import json
from aws_cdk import (
    Stack,
    aws_eks as eks,
    aws_ec2 as ec2,
)
from constructs import Construct
from aws_cdk import Duration
from aws_cdk.lambda_layer_kubectl_v31 import KubectlV31Layer
import boto3
from typing import List
from botocore.exceptions import ClientError

def validate_vpc_config(vpc_config: dict) -> None:
    """
    Validates VPC configuration parameters
    Args:
        vpc_config: Dictionary containing VPC configuration
    Raises:
        ValueError: If any validation fails
    """
    if 'number_of_azs' not in vpc_config:
        raise ValueError("number_of_azs is required in vpc_config")
    
    if vpc_config['number_of_azs'] < 2 or vpc_config['number_of_azs'] > 3:
        raise ValueError("number_of_azs must be between 2 and 3")
    
    if 'subnet_cidr_masks' not in vpc_config:
        raise ValueError("subnet_cidr_masks is required in vpc_config")
        
    if 'public' not in vpc_config['subnet_cidr_masks']:
        raise ValueError("public subnet CIDR mask is required")
        
    if 'private' not in vpc_config['subnet_cidr_masks']:
        raise ValueError("private subnet CIDR mask is required")
    
    for subnet_type, mask in vpc_config['subnet_cidr_masks'].items():
        if mask < 16 or mask > 28:
            raise ValueError(f"CIDR mask for {subnet_type} subnets must be between 16 and 28")

def get_supported_k8s_versions() -> List[str]:
    """
    Gets the list of supported Kubernetes versions from EKS API using describe_cluster_versions
    Returns:
        List[str]: List of supported Kubernetes versions
    """
    try:
        eks_client = boto3.client('eks')
        response = eks_client.describe_cluster_versions()
        
        # Extract versions from the response
        versions = []
        for version in response['clusterVersions']:
            versions.append(version['clusterVersion'])
            
        return sorted(versions)
    except ClientError as e:
        raise ValueError(f"Failed to get supported Kubernetes versions: {str(e)}")

def validate_kubernetes_version(version: str) -> bool:
    """
    Validates if the provided Kubernetes version is supported by EKS
    Args:
        version: Kubernetes version string (e.g., "1.28")
    Returns:
        bool: True if version is supported, False otherwise
    """
    supported_versions = get_supported_k8s_versions()
    
    # Handle cases where version might include patch number (e.g., "1.28.0")
    base_version = '.'.join(version.split('.')[:2])
    
    # Check if the base version exists in any of the supported versions
    return any(supported_version.startswith(base_version) for supported_version in supported_versions)

def validate_node_group_params(params: dict) -> None:
    """
    Validates the node group parameters for EKS cluster creation
    Args:
        params: Dictionary containing cluster and node group parameters
    Raises:
        ValueError: If any validation fails
    """
    required_fields = ['name', 'instance_type', 'min_size', 'max_size', 'desired_size', 'disk_size']
    
    if 'cluster_name' not in params:
        raise ValueError("cluster_name is required in parameters")
    
    if 'kubernetes_version' not in params:
        raise ValueError("kubernetes_version is required in parameters")
    
    if 'node_groups' not in params or not params['node_groups']:
        raise ValueError("At least one node group configuration is required")
    
    for ng in params['node_groups']:
        missing_fields = [field for field in required_fields if field not in ng]
        if missing_fields:
            raise ValueError(f"Missing required fields for node group: {missing_fields}")
        
        # Validate instance types for accelerator nodes
        labels = ng.get('labels', {})
        if 'accelerator' in labels:
            accelerator_type = labels['accelerator']
            instance_type = ng['instance_type']
            
            if accelerator_type == 'nvidia':
                if not instance_type.startswith(('g5', 'p3', 'p4')):
                    raise ValueError(
                        f"Node group {ng['name']} is labeled for NVIDIA but instance type "
                        f"{instance_type} is not a GPU instance type. Use g5, p3, or p4 instances."
                    )
            elif accelerator_type == 'inferentia':
                if not instance_type.startswith(('inf1', 'inf2')):
                    raise ValueError(
                        f"Node group {ng['name']} is labeled for Inferentia but instance type "
                        f"{instance_type} is not an Inferentia instance type. Use inf1 or inf2 instances."
                    )
        
        if ng['min_size'] > ng['max_size']:
            raise ValueError(
                f"min_size ({ng['min_size']}) cannot be greater than "
                f"max_size ({ng['max_size']}) for node group {ng['name']}"
            )
        
        if ng['desired_size'] < ng['min_size'] or ng['desired_size'] > ng['max_size']:
            raise ValueError(
                f"desired_size ({ng['desired_size']}) must be between "
                f"min_size ({ng['min_size']}) and max_size ({ng['max_size']}) "
                f"for node group {ng['name']}"
            )
        
        if ng['disk_size'] < 20:
            raise ValueError(
                f"disk_size for node group {ng['name']} must be at least 20GB"
            )

class LaunchpadStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, cluster_parameters: dict, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        # print_available_ami_types()
        if 'vpc_config' not in cluster_parameters:
            raise ValueError("vpc_config is required in parameters")
        
        vpc_config = cluster_parameters['vpc_config']
        validate_vpc_config(vpc_config)
        
        # Load parameters from JSON file
        with open('cluster_parameters.json', 'r') as f:
            params = json.load(f)

        validate_node_group_params(params)

        vpc = ec2.Vpc(
            self, "InferenceVPC",
            max_azs=vpc_config['number_of_azs'],
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=vpc_config['subnet_cidr_masks']['private']
                ),
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=vpc_config['subnet_cidr_masks']['public']
                )
            ]
        )

        # Create EKS Cluster
        cluster = eks.Cluster(
            self, 
            "InferenceCluster",
            cluster_name=cluster_parameters['cluster_name'],
            version=eks.KubernetesVersion.of(cluster_parameters['kubernetes_version']),
            kubectl_layer=KubectlV31Layer(self, "KubectlLayer"),
            default_capacity=0,
            vpc=vpc,
            vpc_subnets=[ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)],
            endpoint_access=eks.EndpointAccess.PUBLIC_AND_PRIVATE,
            authentication_mode=eks.AuthenticationMode.API_AND_CONFIG_MAP
        )

        # Add managed node groups based on parameters
        accelerators = set()
        for ng_params in cluster_parameters['node_groups']:
            taints = []
            if 'taints' in ng_params:
                for taint in ng_params['taints']:
                    taints.append(eks.TaintSpec(
                        key=taint['key'],
                        value=taint['value'],
                        effect=eks.TaintEffect(taint['effect'])
                    ))
            
            # Get labels with empty dict as default
            labels = ng_params.get('labels', {})
            
            # Debug prints
            print(f"\nProcessing node group: {ng_params['name']}")
            print(f"Instance type: {ng_params['instance_type']}")
            print(f"Labels: {labels}")
            
            # Track accelerator types for plugin installation
            if 'accelerator' in labels:
                accelerators.add(labels['accelerator'])
                print(f"Detected accelerator: {labels['accelerator']}")

            # Get the appropriate AMI type
            ami_type = self._get_ami_type(ng_params['instance_type'], labels)
            
            print(f"Selected AMI type: {ami_type}")
            cluster.add_nodegroup_capacity(
                ng_params['name'],
                instance_types=[ec2.InstanceType(ng_params['instance_type'])],
                min_size=ng_params['min_size'],
                max_size=ng_params['max_size'],
                desired_size=ng_params['desired_size'],
                disk_size=ng_params['disk_size'],
                labels=ng_params.get('labels', {}),
                taints=taints,
                ami_type=ami_type,
                subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)
            )

        # if 'nvidia' in accelerators:
        #     cluster.add_helm_chart(
        #     "nvidia-device-plugin",
        #     chart="nvidia-device-plugin",
        #     repository="https://nvidia.github.io/k8s-device-plugin",
        #     namespace="nvidia-device-plugin",
        #     create_namespace=True,
        #     values={
        #         "tolerations": [{
        #         "key": "nvidia.com/gpu",
        #         "operator": "Exists",
        #         "effect": "NoSchedule"
        #     }],
        #         "nodeSelector": {
        #             "accelerator": "nvidia"
        #         }
        #     },
        #     wait=True,
        #     timeout=Duration.minutes(10)
        # )

        # Install Neuron device plugin if Inferentia nodes are present
        # if 'inferentia' in accelerators:
        #     cluster.add_helm_chart(
        #     "aws-neuron-device-plugin",
        #     chart="aws-neuron-device-plugin",
        #     repository="oci://public.ecr.aws/neuron/neuron-helm-chart",
        #     wait=True,
        #     timeout=Duration.minutes(10),
        #     values={
        #         "tolerations": [{  # Added toleration for Inferentia nodes
        #             "key": "aws.amazon.com/neuron",
        #             "operator": "Exists",
        #             "effect": "NoSchedule"
        #         }],
        #         "nodeSelector": {
        #             "accelerator": "inferentia"
        #         }
        #     }
        # )
            
    def _get_ami_type(self, instance_type: str, labels: dict) -> eks.NodegroupAmiType:
        """
        Determine the appropriate AMI type based on instance type and labels
        Args:
            instance_type: EC2 instance type
            labels: Node group labels
        Returns:
            NodegroupAmiType: The appropriate AMI type for the instance
        """
        print(f"Selecting AMI type for instance {instance_type} with labels {labels}")  # Debug print
        
        # Check if this is an accelerator node
        if 'accelerator' in labels:
            accelerator_type = labels['accelerator']
            if accelerator_type == 'nvidia':
                return eks.NodegroupAmiType.AL2_X86_64_GPU
            elif accelerator_type == 'inferentia':
                return eks.NodegroupAmiType.AL2023_X86_64_NEURON
        
        # Default to AL2 for non-accelerator nodes
        return eks.NodegroupAmiType.AL2_X86_64
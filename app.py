#!/usr/bin/env python3
import os
import json
import aws_cdk as cdk

from launchpad.launchpad_stack import LaunchpadStack


app = cdk.App()

params_file = app.node.try_get_context('params_file') or 'cluster_parameters.json'
if not os.path.exists(params_file):
    raise ValueError(f"Parameters file {params_file} does not exist")

# Load parameters from JSON file
with open('cluster_parameters.json', 'r') as f:
    cluster_parameters = json.load(f)

LaunchpadStack(app, "LaunchpadStack", cluster_parameters=cluster_parameters)


app.synth()

import aws_cdk as core
import aws_cdk.assertions as assertions

from launchpad.launchpad_stack import LaunchpadStack

# example tests. To run these tests, uncomment this file along with the example
# resource in launchpad/launchpad_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = LaunchpadStack(app, "launchpad")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })

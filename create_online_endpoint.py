from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import ManagedOnlineEndpoint
import datetime


# Get a handle to workspace
try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    print('Fallback!')
    credential = InteractiveBrowserCredential()

ml_client = MLClient.from_config(credential=credential)

# create an online endpoint
online_endpoint_name = "tr1-endpoint-" + datetime.datetime.now().strftime("%m%d%H%M%f")
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Online endpoint for Transformers",
    auth_mode="key",
)

print(ml_client.begin_create_or_update(endpoint).result())

# create a blue deployment
model = Model(
    path="codeparrot-ds/",
    type=AssetTypes.CUSTOM_MODEL,
    description="my sample mlflow model",
)

blue_deployment = ManagedOnlineDeployment(
    name="bluetransformers",
    endpoint_name=online_endpoint_name,
    model=model,
    instance_type="Standard_F4s_v2",
    instance_count=1,
)

print(ml_client.online_deployments.begin_create_or_update(blue_deployment).result())
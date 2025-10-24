from dotenv import load_dotenv
from mlflow.deployments import get_deploy_client
from mlflow.genai.scorers import Guidelines
from mlflow.genai.judges import is_correct
from typing import Dict, Any
import pandas as pd
import mlflow

import os

# Load Environment Variables
load_dotenv()

# setup
gateway_uri = os.getenv("MLFLOW_GATEWAY_URI")
gateway = get_deploy_client(gateway_uri)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)
mlflow.autolog()

# create experiment
def create_experiment(experiment_name):
    experiment_id = mlflow.create_experiment(name=experiment_name)
    return experiment_id

# register prompt
def register_or_update_prompt(prompt_id, prompt_template, commit_message):
    updated_prompt = mlflow.genai.register_prompt(
        name=prompt_id,
        template=prompt_template,
        commit_message=commit_message
    )
    return updated_prompt

@mlflow.trace
def generate_from_prompt(prompt):
    data = dict(messages=[{"role": "user", "content": prompt}])
    return gateway.predict(endpoint="chat", inputs=data)

def run_experiment_without_dataset(experiment_id, prompt_id, prompt_version, run_id, **params):
    with mlflow.start_run(run_name=run_id, experiment_id=experiment_id) as run:
        mlflow.log_params(params)
        prompt_template = mlflow.genai.load_prompt(f"prompts:/{prompt_id}/{prompt_version}")
        prompt = prompt_template.format(**params)
        result = generate_from_prompt(prompt)
        mlflow.log_text(str(result), "result.txt")
        response = result['choices'][0]['message']['content']
        return response

def create_evaluation_dataset(experiment_id, dataset_name, df):
    dataset = mlflow.genai.datasets.create_dataset(
        experiment_id=experiment_id,
        name=dataset_name
    )
    # Add the data from the DataFrame to the dataset
    dataset.merge_records(df)
    return dataset

def pred_fn(adjective, noun):
    params = {
        "adjective": adjective,
        "noun": noun        
    }
    prompt_template = mlflow.genai.load_prompt(f"prompts:/{prompt_id}/{prompt_version}")
    prompt = prompt_template.format(**params)
    return generate_from_prompt(prompt)

def get_scorer():
    judge_model_uri = "openai:/gpt-3.5-turbo"
    custom_guidelines = [
        "The response must be in a friendly and informative tone.",
        "The response must be under 30 words."
        ]
    guidelines_scorer = Guidelines(
        model=judge_model_uri,
        guidelines=custom_guidelines
        )
    return guidelines_scorer

def run_experiment_with_dataset(run_id, experiment_id, prompt_id, prompt_version, dataset_id): 
    with mlflow.start_run(run_name=run_id, experiment_id=experiment_id) as run:
        dataset = mlflow.genai.datasets.get_dataset(dataset_id=dataset_id)
        # transform the records in the dataset
        scorer = get_scorer()
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=pred_fn,
            scorers=[scorer],
        )
        return results


# # ==============================================================================
# # 1. CREATE EXPERIMENT
# # ==============================================================================
# experiment_id = create_experiment("Demo_Experiment")
# print(f'Created experiment with ID: {experiment_id}')
experiment_id = "1"

# # ==============================================================================
# # 2. PROMPT MANAGEMENT
# # ==============================================================================
prompt_id = "insult-generator-prompt"
# prompt_template = "Come up with a creative, funny, and light-hearted insult about a {{adjective}} {{noun}}."
# prompt = register_or_update_prompt(
#     prompt_id=prompt_id,
#     prompt_template=prompt_template,
#     commit_message="Initial version of the insult generator prompt."
# )
# print(prompt)
# prompt_version = prompt.version
prompt_version = "1"

# # ==============================================================================
# # 3. RUN EXPERIMENT WITHOUT DATASET
# # # ==============================================================================
# params = {
#     "adjective": "clumsy",
#     "noun": "pigeon"
# }
# result = run_experiment_without_dataset(
#     experiment_id=experiment_id,
#     run_id="Insult_Run_Without_Dataset_12",
#     prompt_id=prompt_id,
#     prompt_version=prompt_version,
#     **params
# )
# print(f"Generated Insult without Dataset: {result}")

# # ==============================================================================
# # 4. DATASET
# # ==============================================================================
# # Create a simple evaluation dataset
# "adjective": ["sleepy", "silly", "fluffy"],
#         "noun": ["sloth", "penguin", "kitten"],
#         "ground_truth": [
#             "You move with the urgency of a sleepy sloth on a tranquilizer.",
#             "You have the directional sense of a silly penguin in a desert.",
#             "You're about as threatening as a fluffy kitten in a marshmallow factory."
#         ],
# eval_data = pd.DataFrame(
#     [
#     {
#         "inputs": { "adjective": "sleepy", "noun": "sloth" },
#         "expectations" : {"expected_facts" : ["response should be an insult", "insult should be light-hearted"]},
#     },
#     {
#         "inputs": { "adjective": "silly", "noun": "penguin" },
#         "expectations" : {"expected_facts" : ["response should be an insult", "insult should be light-hearted"]},
#     },
#     {
#         "inputs": { "adjective": "fluffy", "noun": "kitten" },
#         "expectations" : {"expected_facts" : ["response should be an insult", "insult should be light-hearted"]}
#     }]
# )
# print(f"Created evaluation DataFrame with {len(eval_data)} records.")
# dataset_name = "insult-eval-dataset_v1"
# create_dataset_result = create_evaluation_dataset(
#     experiment_id=experiment_id,
#     dataset_name=dataset_name,
#     df=eval_data)
# print(f"Created evaluation dataset: {create_dataset_result.dataset_id}")
# dataset_id = create_dataset_result.dataset_id
dataset_id = "d-1ca7b925e6b047d6bfedeb22bbe8b5a1"
run_exp_res = run_experiment_with_dataset(
    run_id="Insult_Run_With_Dataset_1",
    experiment_id=experiment_id,
    prompt_id=prompt_id,
    prompt_version=prompt_version,
    dataset_id=dataset_id
)
# print(run_exp_res)
import os
import git
import openai
import json
import torch
import requests
from transformers import pipeline
from typing import List, Dict, Union

class CodeReviewAgent:
    def __init__(self, repo_path: str, 
                 branch_name: str,
                 master_branch_name: str = "main",
                 llm_model: str = "gpt-4", 
                 output_file: str = "code_review.txt", 
                 provider: str = "openai", 
                 max_tokens: int = 2048, 
                 temperature: float = 0.1, 
                 openai_api_key: str = "", 
                 hf_api_key: str = ""):  
        self.repo_path = repo_path
        self.branch_name = branch_name
        self.llm_model = llm_model
        self.output_file = output_file
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.openai_api_key = openai_api_key
        self.hf_api_key = hf_api_key
        self.repo = git.Repo(repo_path)
        self.master_branch = master_branch_name
        self.device = 0 if torch.cuda.is_available() else -1

        if self.provider == "openai":
            self.openai_client = openai.Client(api_key=self.openai_api_key)
        elif self.provider == "local":
            self.local_llm = pipeline("text-generation", model=self.llm_model, device=self.device)

    def get_diff(self) -> str:
        """Retrieve diff between the branch and master."""
        self.repo.git.checkout(self.branch_name)
        diff = self.repo.git.diff(f"{self.master_branch}..{self.branch_name}", unified=3)
        return diff

    def analyze_with_llm(self, diff_text: str) -> List[Dict[str, Union[str, int]]]:
        """Use LLM to analyze the code changes and return structured feedback."""
        prompt = f"""
        You are an experienced software engineer. Review the following code changes and provide structured feedback in JSON format.
        Format your response strictly as a list of dictionaries, where each dictionary contains the following keys:
        - 'file': the filename (if available)
        - 'line': the line number of the comment (use 'N/A' if not applicable)
        - 'comment': your constructive feedback

        For example:
        [   
            {{'file': 'app/main.py', 'line': 1, 'comment': 'Your comment goes here.'}},
            {{'file': 'app/utils.py', 'line': 21, 'comment': 'Another coment goes here.'}}
        ]

        Code Changes:
        {diff_text}
        """
        
        try:
            if self.provider == "openai":
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "system", "content": "You are a code review assistant."},
                              {"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                llm_output = response.choices[0].message.content
            elif self.provider == "local":
                response = self.local_llm(prompt, max_length=self.max_tokens, temperature=self.temperature, truncation=True)
                llm_output = response[0]["generated_text"]
            elif self.provider == "huggingface":
                headers = {"Authorization": f"Bearer {self.hf_api_key}"}
                payload = {"inputs": prompt, "parameters": {"max_new_tokens": self.max_tokens, "temperature": self.temperature}}
                hf_response = requests.post(f"https://api-inference.huggingface.co/models/{self.llm_model}", headers=headers, json=payload)
                print(hf_response)
                if hf_response.status_code == 200:
                    llm_output = hf_response.json()[0]["generated_text"][len(prompt):]
                else:
                    print(f"Error: {hf_response.status_code}, {hf_response.text}")
                    llm_output = ""
            else:
                raise ValueError("Unsupported LLM provider")
            
            if llm_output:
                try:
                    llm_output = json.loads(llm_output)
                    return llm_output
                except json.JSONDecodeError as e:
                    return [{'comment': llm_output}]
            else:
                print("No review generated.")
                return []

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing LLM response: {e}")
            return []

    def save_review(self, review_comments: List[Dict[str, Union[str, int]]]):
        """Save review comments to a file in a structured format."""
        with open(self.output_file, "w") as file:
            for comment in review_comments:
                file.write(f"File: {comment.get('file', 'N/A')}\n")
                file.write(f"Line: {comment.get('line', 'N/A')}\n")
                file.write(f"Comment: {comment.get('comment')}\n")
                file.write("-" * 100 + "\n")
        print(f"Review saved to {self.output_file}")

    def run_review(self):
        """Execute the code review process."""
        diff_text = self.get_diff()
        review_comments = self.analyze_with_llm(diff_text)
        self.save_review(review_comments)

if __name__ == "__main__":
    repo_path = "./sample_repo"
    branch_name = "feture_branch"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    agent = CodeReviewAgent(repo_path, branch_name, provider="huggingface", llm_model="tiiuae/falcon-7b-instruct", hf_api_key=hf_api_key)
    agent.run_review()

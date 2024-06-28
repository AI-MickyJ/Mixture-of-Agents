import os
import asyncio
from typing import List, Dict
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai
from dotenv import load_dotenv
from termcolor import colored
import html

load_dotenv()

class Agent:
    def __init__(self, name: str):
        self.name = name

    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

class GPT4Agent(Agent):
    def __init__(self):
        super().__init__("GPT-4")
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content

class ClaudeAgent(Agent):
    def __init__(self):
        super().__init__("Claude")
        self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def generate(self, prompt: str) -> str:
        response = await self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text

class GeminiAgent(Agent):
    def __init__(self):
        super().__init__("Gemini")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    async def generate(self, prompt: str) -> str:
        response = await asyncio.to_thread(self.model.generate_content, prompt)
        return response.text

class MixtureOfAgents:
    def __init__(self, agents: List[Agent], num_layers: int):
        self.agents = agents
        self.num_layers = num_layers
        self.claude_agent = ClaudeAgent()  # For aggregation

    async def aggregate_responses(self, responses: List[str]) -> str:
        aggregation_prompt = f"Synthesize the following responses into a single, coherent answer:\n\n" + "\n\n".join(responses)
        return await self.claude_agent.generate(aggregation_prompt)

    async def process_layer(self, prompt: str) -> tuple:
        tasks = [agent.generate(prompt) for agent in self.agents]
        responses = await asyncio.gather(*tasks)
        aggregated = await self.aggregate_responses(responses)
        return responses, aggregated

    async def generate(self, prompt: str) -> tuple:
        current_prompt = prompt
        all_responses = []
        for _ in range(self.num_layers):
            responses, current_prompt = await self.process_layer(current_prompt)
            all_responses.append(responses)
        
        final_rewrite_prompt = f"Rewrite the following response to ensure clarity and coherence for a user who hasn't seen the intermediate steps. The response should directly address the original prompt: '{prompt}'\n\nResponse to rewrite:\n{current_prompt}"
        final_response = await self.claude_agent.generate(final_rewrite_prompt)
        
        return all_responses, final_response

def generate_html_report(prompt: str, all_responses: List[List[str]], final_response: str) -> str:
    html_content = f"""
    <html>
    <head>
        <title>MoA Response Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
            h1, h2 {{ color: #333; }}
            .agent-response {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; }}
            .agent-1 {{ background-color: #ffeeee; }}
            .agent-2 {{ background-color: #eeffee; }}
            .agent-3 {{ background-color: #eeeeff; }}
            .final-response {{ background-color: #e6f3ff; padding: 15px; border: 2px solid #b3d9ff; }}
        </style>
    </head>
    <body>
        <h1>MoA Response Report</h1>
        <h2>Original Prompt:</h2>
        <p>{html.escape(prompt)}</p>
        
        <h2>Intermediate Outputs:</h2>
    """
    
    for layer, responses in enumerate(all_responses, 1):
        html_content += f"<h3>Layer {layer}:</h3>"
        for i, response in enumerate(responses, 1):
            html_content += f"""
            <div class="agent-response agent-{i}">
                <h4>Agent {i}:</h4>
                <p>{html.escape(response)}</p>
            </div>
            """
    
    html_content += f"""
        <h2>Final MoA Response:</h2>
        <div class="final-response">
            <p>{html.escape(final_response)}</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

async def interactive_session(moa: MixtureOfAgents):
    print("Welcome to the Mixture of Agents (MoA) interactive session!")
    print("Type your questions or prompts, and the MoA model will respond.")
    print("To exit, type 'quit', 'exit', or 'q'.")
    
    while True:
        user_input = input("\nEnter your prompt: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the MoA model. Goodbye!")
            break
        
        print("Processing your request. This may take a moment...")
        all_responses, final_response = await moa.generate(user_input)
        
        print("\nIntermediate outputs:")
        for layer, responses in enumerate(all_responses, 1):
            print(f"\nLayer {layer}:")
            for i, response in enumerate(responses):
                color = ['red', 'green', 'yellow'][i % 3]
                print(colored(f"Agent {i+1}:", color))
                print(colored(response, color))
                print()
        
        print("\nFinal MoA Response:")
        print(colored(final_response, 'cyan'))
        
        # Generate and save HTML report
        html_content = generate_html_report(user_input, all_responses, final_response)
        with open("moa_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("\nDetailed HTML report saved as 'moa_report.html'")

async def main():
    # Initialize agents
    gpt4_agent = GPT4Agent()
    claude_agent = ClaudeAgent()
    gemini_agent = GeminiAgent()

    # Create MixtureOfAgents with 3 layers
    moa = MixtureOfAgents([gpt4_agent, claude_agent, gemini_agent], num_layers=3)

    # Start interactive session
    await interactive_session(moa)

if __name__ == "__main__":
    asyncio.run(main())
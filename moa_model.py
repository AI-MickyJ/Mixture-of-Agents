import os
import asyncio
from typing import List, Dict
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai
from dotenv import load_dotenv
from termcolor import colored
import html
import re
from difflib import SequenceMatcher

load_dotenv()

class Agent:
    def __init__(self, name: str, model: str, role: str):
        self.name = name
        self.model = model
        self.role = role

    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

class GPT4Agent(Agent):
    def __init__(self):
        super().__init__("GPT-4", "gpt-4o", "Agent 1")
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
        super().__init__("Claude", "claude-3-sonnet-20240229", "Agent 2")
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
        super().__init__("Gemini Pro 1.5", "gemini-1.5-pro", "Agent 3")
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

    async def process_layer(self, prompt: str, layer_num: int) -> tuple:
        print(f"Starting layer {layer_num} pass...")
        tasks = [agent.generate(prompt) for agent in self.agents]
        responses = await asyncio.gather(*tasks)
        aggregated = await self.aggregate_responses(responses)
        print(f"Layer {layer_num} complete.")
        print(f"Aggregate response for layer {layer_num}:")
        print(colored(aggregated, 'magenta'))
        print()
        return responses, aggregated

    async def generate(self, prompt: str) -> tuple:
        current_prompt = prompt
        all_responses = []
        for i in range(self.num_layers):
            responses, current_prompt = await self.process_layer(current_prompt, i+1)
            all_responses.append(responses)
        
        print("All layers complete. Generating final response...")
        final_compilation_prompt = f"""
        Compile the best parts of the following responses into a comprehensive, coherent answer that directly addresses the original prompt. 
        Ensure that the final response is detailed, well-structured, and incorporates the most valuable insights from each input.
        The response should fully answer all aspects of the original prompt and maintain a high level of quality and coherence.
        Original prompt: '{prompt}'

        Responses to compile:
        {current_prompt}
        """
        final_response = await self.claude_agent.generate(final_compilation_prompt)
        
        # Calculate utilization percentages
        utilization = self.calculate_utilization(all_responses, final_response)
        
        return all_responses, final_response, utilization

    def calculate_utilization(self, all_responses: List[List[str]], final_response: str) -> Dict[str, float]:
        utilization = {}
        for i, agent in enumerate(self.agents):
            agent_responses = [layer[i] for layer in all_responses]
            combined_response = " ".join(agent_responses)
            similarity = SequenceMatcher(None, combined_response, final_response).ratio()
            utilization[agent.name] = similarity * 100
        total = sum(utilization.values())
        return {k: v / total * 100 for k, v in utilization.items()}

def generate_html_report(prompt: str, all_responses: List[List[str]], final_response: str, utilization: Dict[str, float], agents: List[Agent]) -> str:
    html_content = f"""
    <html>
    <head>
        <title>MoA Response Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .agent-response {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; }}
            .agent-1 {{ background-color: #ffeeee; }}
            .agent-2 {{ background-color: #eeffee; }}
            .agent-3 {{ background-color: #eeeeff; }}
            .final-response {{ background-color: #e6f3ff; padding: 15px; border: 2px solid #b3d9ff; }}
            .layer {{ margin-bottom: 30px; padding: 15px; border: 1px solid #999; }}
        </style>
    </head>
    <body>
        <h1>MoA Response Report</h1>
        <h2>Original Prompt:</h2>
        <p>{html.escape(prompt)}</p>

        <h2>Final MoA Response:</h2>
        <div class="final-response">
            <p>{html.escape(final_response)}</p>
        </div>
        
        <h2>Intermediate Outputs:</h2>
    """
    
    for layer, responses in enumerate(all_responses, 1):
        html_content += f'<div class="layer"><h3>Layer {layer}:</h3>'
        for i, response in enumerate(responses):
            agent = agents[i]
            html_content += f"""
            <div class="agent-response agent-{i+1}">
                <h4>{agent.name} ({agent.model}) - {agent.role}:</h4>
                <p>{html.escape(response)}</p>
            </div>
            """
        html_content += '</div>'
    
    html_content += f"""
        <h2>Agent Utilization:</h2>
        <ul>
    """
    
    for agent, percentage in utilization.items():
        html_content += f"<li>{agent}: {percentage:.2f}%</li>"
    
    html_content += """
        </ul>
    </body>
    </html>
    """
    
    return html_content

def generate_markdown_report(prompt: str, all_responses: List[List[str]], final_response: str, utilization: Dict[str, float], agents: List[Agent]) -> str:
    markdown_content = f"""## Original Prompt:
{prompt}

## Final MoA Response:
{final_response}

## Intermediate Outputs:
"""
    
    for layer, responses in enumerate(all_responses, 1):
        markdown_content += f"### Layer {layer}:\n\n"
        for i, response in enumerate(responses):
            agent = agents[i]
            markdown_content += f"#### {agent.name} ({agent.model}) - {agent.role}:\n{response}\n\n"
    
    markdown_content += """## Agent Utilization:\n\n"""
    for agent, percentage in utilization.items():
        markdown_content += f"- {agent}: {percentage:.2f}%\n"
    
    return markdown_content

def sanitize_filename(filename):
    return re.sub(r'[^\w\-_\. ]', '_', filename)

async def interactive_session(moa: MixtureOfAgents):
    print("Welcome to the Mixture of Agents (MoA) interactive session!")
    print(f"The current setup uses {moa.num_layers} layers.")
    print("The recommended default is three layers.")
    
    layers_input = input("Would you like to start with three layers, or a different amount? (Enter Y/Yes for 3 layers, or a number): ").strip().lower()
    
    if layers_input in ['y', 'yes']:
        moa.num_layers = 3
    else:
        try:
            num_layers = int(layers_input)
            if num_layers < 1:
                print("Invalid input. Using the default of 3 layers.")
            else:
                moa.num_layers = num_layers
        except ValueError:
            print("Invalid input. Using the default of 3 layers.")
    
    print(f"\nUsing {moa.num_layers} layers for this session.")
    print("Type your questions or prompts, and the MoA model will respond.")
    print("To exit, type 'quit', 'exit', or 'q'.")
    
    while True:
        user_input = input("\nEnter your prompt: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the MoA model. Goodbye!")
            break
        
        print("\nProcessing your request. This may take some time depending on the number of layers.")
        print("Please be patient while the MoA completes the process.")
        all_responses, final_response, utilization = await moa.generate(user_input)
        
        print("\nFinal MoA Response:")
        print(colored(final_response, 'cyan'))
        
        print("\nAgent Utilization:")
        for agent, percentage in utilization.items():
            print(f"{agent}: {percentage:.2f}%")
        
        # New saving process
        save_final = input("\nDo you want to save the final response? (Y/N): ").strip().lower()
        if save_final in ['y', 'yes']:
            format_pref = input("Do you want this in HTML or MD? ").strip().lower()
            if format_pref in ['html', 'h']:
                format_pref = 'html'
            else:
                format_pref = 'md'
            
            base_filename = sanitize_filename(user_input[:50])
            filename = f"{base_filename}_final_response.{format_pref}"
            
            if format_pref == 'html':
                content = f"<html><body><h1>Final MoA Response</h1><p>{html.escape(final_response)}</p></body></html>"
            else:
                content = f"# Final MoA Response\n\n{final_response}"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Final response saved as '{filename}'")
            
            save_log = input("Do you also want a full log of the agent responses at each layer? (Y/N): ").strip().lower()
            if save_log in ['y', 'yes']:
                log_filename = f"{base_filename}_detailed_log.{format_pref}"
                if format_pref == 'html':
                    content = generate_html_report(user_input, all_responses, final_response, utilization, moa.agents)
                else:
                    content = generate_markdown_report(user_input, all_responses, final_response, utilization, moa.agents)
                
                with open(log_filename, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Detailed log saved as '{log_filename}'")
        else:
            print("No files saved.")

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
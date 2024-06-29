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
            max_tokens=1000        )
        return response.choices[0].message.content

class ClaudeAgent(Agent):
    def __init__(self):
        super().__init__("Claude", "claude-3-sonnet-20240229", "Agent 2")
        self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        response = await self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text

class GeminiAgent(Agent):
    def __init__(self):
        super().__init__("Gemini", "gemini-1.5-pro", "Agent 3")
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

    async def aggregate_responses(self, prompt: str, current_prompt: str, responses: List[str], layer_num: int) -> str:
        aggregation_prompt = f"""
        Original prompt: {prompt}

        Current context:
        {current_prompt}

        Compile the best parts of the following responses into a comprehensive, coherent answer:
        {responses[0]}

        {responses[1]}

        {responses[2]}

        Your task:
        1. Synthesize these responses, preserving the most valuable insights from each.
        2. Ensure your answer is well-structured and formatted, borrowing the best layout ideas from the original responses.
        3. Address all aspects of the original prompt comprehensively.
        4. Maintain focus on answering the original prompt while incorporating the insights from these responses.
        5. Your response should be more detailed and nuanced than previous layers. As this is layer {layer_num}, make sure to provide a {'more' if layer_num > 1 else ''} thorough and in-depth analysis.
        6. Include any new perspectives or deeper insights that have emerged from the combination of previous responses.
        7. If there are any conflicting viewpoints in the responses, address them and provide a balanced perspective.

        Aim to produce a response that is {'even more comprehensive and insightful than the previous layer' if layer_num > 1 else 'comprehensive and insightful'}, building upon the accumulated knowledge and expanding the depth of the analysis."""
        return await self.claude_agent.generate(aggregation_prompt, max_tokens=2000)

    async def process_layer(self, original_prompt: str, current_prompt: str, layer_num: int) -> tuple:
        print(f"Starting layer {layer_num} pass...")
        
        if layer_num == 1:
            layer_prompt = original_prompt
        else:
            layer_prompt = f"""Original prompt: {original_prompt}

    Additional context from previous layers:
    {current_prompt}

    In forming your response, please leverage and review the additional context provided above. Provide a new, more detailed response to the original question:

    "{original_prompt}"

    Utilize the given context and add any further thoughts, arguments, or insights that you feel would be beneficial to answering the original prompt comprehensively. Your response should be more detailed and nuanced than previous layers, incorporating the accumulated knowledge and insights."""

        tasks = [agent.generate(layer_prompt) for agent in self.agents]
        responses = await asyncio.gather(*tasks)
        aggregated = await self.aggregate_responses(original_prompt, current_prompt, responses, layer_num)
        print(f"Layer {layer_num} complete.")
        print(f"Aggregate response for layer {layer_num}:")
        print(colored(aggregated, 'magenta'))
        print()
        return layer_prompt, responses, aggregated

    async def generate(self, prompt: str) -> tuple:
        original_prompt = prompt
        current_prompt = prompt
        all_responses = []
        all_aggregates = []
        all_layer_prompts = []
        for i in range(self.num_layers):
            layer_prompt, responses, aggregated = await self.process_layer(original_prompt, current_prompt, i+1)
            all_responses.append(responses)
            all_aggregates.append(aggregated)
            all_layer_prompts.append(layer_prompt)
            current_prompt = aggregated
        
        print("All layers complete. Generating final response...")
        final_compilation_prompt = f"""Original prompt: {original_prompt}

    Final context: {current_prompt}

    Compile a comprehensive, coherent final answer that directly addresses the original prompt. 
    Ensure that the final response is detailed, well-structured, and incorporates the most valuable insights from all previous layers.
    Use appropriate headings and subheadings to organize the information effectively.
    Maintain a clear and logical flow throughout the response."""

        final_response = await self.claude_agent.generate(final_compilation_prompt, max_tokens=4000)
        
        # Calculate utilization percentages
        utilization = self.calculate_utilization(all_responses, final_response)
        
        return all_layer_prompts, all_responses, all_aggregates, final_response, utilization

    def calculate_utilization(self, all_responses: List[List[str]], final_response: str) -> Dict[str, float]:
        utilization = {}
        for i, agent in enumerate(self.agents):
            agent_responses = [layer[i] for layer in all_responses]
            combined_response = " ".join(agent_responses)
            similarity = SequenceMatcher(None, combined_response, final_response).ratio()
            utilization[agent.name] = similarity * 100
        total = sum(utilization.values())
        return {k: v / total * 100 for k, v in utilization.items()}
def generate_html_report(prompt: str, all_layer_prompts: List[str], all_responses: List[List[str]], all_aggregates: List[str], final_response: str, utilization: Dict[str, float], agents: List[Agent]) -> str:
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
            .layer-prompt {{ background-color: #f0f0f0; padding: 10px; margin-bottom: 10px; }}
            .layer-aggregate {{ background-color: #fff0f5; padding: 10px; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <h1>MoA Response Report</h1>
        <h2>Original Prompt:</h2>
        <p>{html.escape(prompt)}</p>

        <h2>Agent Utilization:</h2>
        <ul>
    """
    for agent, percentage in utilization.items():
        html_content += f"<li>{agent}: {percentage:.2f}%</li>"
    
    html_content += f"""
        </ul>

        <h2>Final MoA Response:</h2>
        <div class="final-response">
            <p>{html.escape(final_response)}</p>
        </div>
        
        <h2>Intermediate Outputs:</h2>
    """
    
    for layer, (layer_prompt, responses, aggregate) in enumerate(zip(all_layer_prompts, all_responses, all_aggregates), 1):
        html_content += f'<div class="layer"><h3>Layer {layer}:</h3>'
        html_content += f'<div class="layer-prompt"><h4>Layer Prompt:</h4><p>{html.escape(layer_prompt)}</p></div>'
        for i, response in enumerate(responses):
            if i == 2:  # Gemini is always Agent 3
                agent_display = "Gemini (gemini-1.5-pro) - Agent 3"
            else:
                agent = agents[i]
                agent_display = f"{agent.name} ({agent.model}) - {agent.role}"
            html_content += f"""
            <div class="agent-response agent-{i+1}">
                <h4>{html.escape(agent_display)}:</h4>
                <p>{html.escape(response)}</p>
            </div>
            """
        html_content += f'<div class="layer-aggregate"><h4>Layer {layer} Aggregate:</h4><p>{html.escape(aggregate)}</p></div>'
        html_content += '</div>'

    html_content += """
    </body>
    </html>
    """
    
    return html_content

def generate_final_response_html(prompt: str, final_response: str, utilization: Dict[str, float]) -> str:
    html_content = f"""
    <html>
    <head>
        <title>MoA Final Response</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
            h1, h2 {{ color: #333; }}
            .section {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; }}
            .utilization {{ background-color: #f0f0f0; }}
            .final-response {{ background-color: #e6f3ff; }}
        </style>
    </head>
    <body>
        <h1>MoA Final Response</h1>
        
        <div class="section">
            <h2>Original Prompt:</h2>
            <p>{html.escape(prompt)}</p>
        </div>

        <div class="section utilization">
            <h2>Agent Utilization:</h2>
            <ul>
    """
    for agent, percentage in utilization.items():
        html_content += f"<li>{agent}: {percentage:.2f}%</li>"
    
    html_content += f"""
            </ul>
        </div>

        <div class="section final-response">
            <h2>Final MoA Response:</h2>
            <p>{html.escape(final_response)}</p>
        </div>
    </body>
    </html>
    """
    return html_content

def generate_markdown_report(prompt: str, all_layer_prompts: List[str], all_responses: List[List[str]], all_aggregates: List[str], final_response: str, utilization: Dict[str, float], agents: List[Agent]) -> str:
    markdown_content = f"""# MoA Response Report

## Original Prompt:
{prompt}

## Agent Utilization:
"""
    for agent, percentage in utilization.items():
        markdown_content += f"- {agent}: {percentage:.2f}%\n"

    markdown_content += f"""
## Final MoA Response:
{final_response}

## Intermediate Outputs:
"""
    
    for layer, (layer_prompt, responses, aggregate) in enumerate(zip(all_layer_prompts, all_responses, all_aggregates), 1):
        markdown_content += f"### Layer {layer}:\n\n"
        markdown_content += f"#### Layer Prompt:\n{layer_prompt}\n\n"
        for i, response in enumerate(responses):
            if i == 2:  # Gemini is always Agent 3
                agent_display = "Gemini (gemini-1.5-pro) - Agent 3"
            else:
                agent = agents[i]
                agent_display = f"{agent.name} ({agent.model}) - {agent.role}"
            markdown_content += f"#### {agent_display}:\n{response}\n\n"
        markdown_content += f"#### Layer {layer} Aggregate:\n{aggregate}\n\n"

    return markdown_content

def generate_final_response_markdown(prompt: str, final_response: str, utilization: Dict[str, float]) -> str:
    markdown_content = f"""# MoA Final Response

## Original Prompt:
{prompt}

## Agent Utilization:
"""
    for agent, percentage in utilization.items():
        markdown_content += f"- {agent}: {percentage:.2f}%\n"

    markdown_content += f"""
## Final MoA Response:
{final_response}
"""
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
                moa.num_layers = 3
            else:
                moa.num_layers = num_layers
        except ValueError:
            print("Invalid input. Using the default of 3 layers.")
            moa.num_layers = 3
    
    print(f"\nUsing {moa.num_layers} layers for this session.")
    print("Type your questions or prompts, and the MoA model will respond.")
    print("To exit, type 'quit', 'exit', or 'q'.")
    
    while True:
        user_input = input("\nEnter your prompt: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the MoA model. Goodbye!")
            break
        
        print("\nProcessing your request. This may take some time depending on the number of layers.")
        print("Please be patient while the MoA completes the process.")
        all_layer_prompts, all_responses, all_aggregates, final_response, utilization = await moa.generate(user_input)
        
        print("\nFinal MoA Response:")
        print(colored(final_response, 'cyan'))
        
        print("\nAgent Utilization:")
        for agent, percentage in utilization.items():
            print(f"{agent}: {percentage:.2f}%")
        
        # New saving process
        save_final = input("\nDo you want to save the final response? (Y/N): ").strip().lower()
        if save_final in ['y', 'yes']:
            format_pref = input("Do you want to save as HTML, MD, or both? (html/md/both): ").strip().lower()
            
            base_filename = sanitize_filename(user_input[:50])
        
            if format_pref in ['html', 'both']:
                html_filename = f"{base_filename}_final_response.html"
                html_content = generate_final_response_html(user_input, final_response, utilization)
                with open(html_filename, "w", encoding="utf-8") as f:
                    f.write(html_content)
                print(f"Final response saved as '{html_filename}'")
        
            if format_pref in ['md', 'both']:
                md_filename = f"{base_filename}_final_response.md"
                md_content = generate_final_response_markdown(user_input, final_response, utilization)
                with open(md_filename, "w", encoding="utf-8") as f:
                    f.write(md_content)
                print(f"Final response saved as '{md_filename}'")
        
        save_log = input("Do you also want a full log of the agent responses at each layer? (Y/N): ").strip().lower()
        if save_log in ['y', 'yes']:
            if format_pref == 'both':
                log_format = 'both'
            else:
                log_format = input("Do you want to save the log as HTML, MD, or both? (html/md/both): ").strip().lower()
            
            if log_format in ['html', 'both']:
                html_log_filename = f"{base_filename}_detailed_log.html"
                html_content = generate_html_report(user_input, all_layer_prompts, all_responses, all_aggregates, final_response, utilization, moa.agents)
                with open(html_log_filename, "w", encoding="utf-8") as f:
                    f.write(html_content)
                print(f"Detailed HTML log saved as '{html_log_filename}'")
            
            if log_format in ['md', 'both']:
                md_log_filename = f"{base_filename}_detailed_log.md"
                md_content = generate_markdown_report(user_input, all_layer_prompts, all_responses, all_aggregates, final_response, utilization, moa.agents)
                with open(md_log_filename, "w", encoding="utf-8") as f:
                    f.write(md_content)
                print(f"Detailed Markdown log saved as '{md_log_filename}'")
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
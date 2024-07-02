import os
import asyncio
from typing import List, Dict
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
from termcolor import colored
from pathlib import Path
import html
import re
from difflib import SequenceMatcher
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)

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

    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content if response.choices else ""

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
        return response.content[0].text if response.content else ""

class GeminiAgent(Agent):
    def __init__(self):
        super().__init__("Gemini", "gemini-1.5-pro", "Agent 3")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        generation_config = GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.7,
            top_p=1,
            top_k=1
        )
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=generation_config
            )
            return response.text if response.text else ""
        except Exception as e:
            print(f"Error in GeminiAgent: {str(e)}")
            return f"Error occurred: {str(e)}"

class MixtureOfAgents:
    def __init__(self, agents: List[Agent], num_layers: int = 2):
        self.agents = agents
        self.num_layers = num_layers
        self.synthesis_agent = GeminiAgent()  # Gemini for layer synthesis
        self.final_agent = ClaudeAgent()  # Claude for final output

    async def generate(self, prompt: str) -> tuple:
        all_layer_prompts = []
        all_responses = []
        all_syntheses = []
        
        current_context = ""
        for i in range(self.num_layers):
            print(colored(f"\n* Layer {i+1} started", "cyan"))
            layer_prompt = self.create_layer_prompt(prompt, current_context, i)
            all_layer_prompts.append(layer_prompt)

            initial_responses = await self.get_initial_responses(layer_prompt)
            print(colored(f"* Layer {i+1} initial responses received", "green"))
            
            aggregations = await self.get_aggregations(prompt, layer_prompt, initial_responses)
            print(colored(f"* Layer {i+1} aggregations completed", "green"))
            
            all_responses.append(initial_responses + aggregations)

            synthesis, devils_advocate = await self.synthesize_responses(prompt, layer_prompt, aggregations)
            all_syntheses.append((synthesis, devils_advocate))
            current_context = f"Synthesis: {synthesis}\n\nDevil's Advocate: {devils_advocate}"
            print(colored(f"* Layer {i+1} synthesis and devil's advocate perspective generated", "green"))
            print(colored(f"* Layer {i+1} completed", "cyan"))

        print(colored("\n* Generating Final Output", "yellow"))
        final_response = await self.generate_final_output(prompt, all_syntheses)
        print(colored("* Final Output Generated", "yellow"))
        
        utilization = self.calculate_utilization(all_responses, final_response)
        
        return all_layer_prompts, all_responses, all_syntheses, final_response, utilization

    def create_layer_prompt(self, original_prompt: str, context: str, layer: int) -> str:
        if layer == 0:
            return original_prompt
        else:
            return f"""Original prompt: {original_prompt}

    Context from previous layer:
    {context}

    Your task:
    1. Independently solve or answer the original prompt using your own logic and reasoning.
    2. Consider the provided context, but do not be overly influenced by it.
    3. If the context causes you to rethink your instinctive response, explain why and provide your rationale.
    4. Provide a clear, well-reasoned answer to the original prompt."""

    async def get_initial_responses(self, prompt: str) -> List[str]:
        responses = []
        for index, agent in enumerate(self.agents, 1):
            response = await agent.generate(prompt)
            responses.append(response)
            print(colored(f"* Agent {index} initial response received", "green"))
        return responses

    async def get_aggregations(self, original_prompt: str, current_prompt: str, responses: List[str]) -> List[str]:
        aggregations = []
        for i, agent in enumerate(self.agents):
            aggregation_prompt = f"""Original prompt: {original_prompt}

        Current context: {current_prompt}

        Responses from agents:
        {chr(10).join(responses)}

        Your tasks:
        1. Critically analyze and test the logic and reasoning of all responses, including your own.
        2. Identify and challenge ALL assumptions made in the responses, even seemingly obvious ones.
        3. If applicable, perform mathematical verifications to check the validity of the answers.
        4. Explore multiple interpretations of the prompt using a logic tree, considering all possible scenarios.
        5. Conduct a peer review by critically examining and potentially correcting the reasoning of other agents.
        6. Provide your own independent answer to the original prompt, using the responses as context but relying primarily on your own reasoning.
        7. Explain your thought process, including why you agree or disagree with other responses.
        8. If you're unsure about any aspect, state so and explain why.

        Ensure your response is structured, addressing each of these points separately."""

            aggregation = await agent.generate(aggregation_prompt)
            aggregations.append(aggregation)
            print(colored(f"* Agent {i+1} aggregation completed", "green"))
        return aggregations

    async def synthesize_responses(self, original_prompt: str, current_prompt: str, responses: List[str]) -> tuple:
        print(colored("* Starting synthesis and devil's advocate generation", "magenta"))
        synthesis_prompt = f"""Original prompt: {original_prompt}

        Current context: {current_prompt}

        Aggregated responses:
        {chr(10).join(responses)}

        Your tasks:
        1. Assess which of the aggregated responses is the most well-reasoned and logically sound.
        2. Provide a summary of the key points and any disagreements among the responses.
        3. Create a synthesis that will serve as context for the next layer, encouraging further independent analysis.
        4. Include the original prompt in your synthesis.
        5. Highlight areas that may need further consideration or clarification.

        After completing the above tasks, take on an aggressive devil's advocate role:
        6. Challenge the prevailing or most popular answer among the responses.
        7. Critically examine and question basic counting, logic, and assumptions made by all agents.
        8. Identify potential flaws, overlooked aspects, or alternative interpretations that could invalidate the current reasoning.
        9. Propose at least one alternative perspective or solution that hasn't been considered.
        10. If applicable, point out any mathematical or logical inconsistencies in the reasoning.

        Provide your response in two clearly labeled sections: 'Synthesis' and 'Devil's Advocate'."""

        combined_response = await self.synthesis_agent.generate(synthesis_prompt)
        
        # Split the combined response into synthesis and devil's advocate parts
        parts = combined_response.split("Devil's Advocate:", 1)
        synthesis = parts[0].replace("Synthesis:", "").strip()
        
        if len(parts) > 1:
            devils_advocate = parts[1].strip()
        else:
            devils_advocate = "No specific Devil's Advocate perspective was provided. Consider potential alternative viewpoints or challenges to the synthesis."
        
        print(colored("* Synthesis and devil's advocate generation completed", "magenta"))
        return synthesis, devils_advocate

    async def generate_final_output(self, original_prompt: str, all_syntheses: List[tuple]) -> str:
        consolidated_syntheses = "\n\n".join([
            f"Layer {i+1} Synthesis:\n{synthesis}\n\nLayer {i+1} Devil's Advocate:\n{devils_advocate}"
            for i, (synthesis, devils_advocate) in enumerate(all_syntheses)
        ])
        
        final_prompt = f"""Original prompt: {original_prompt}

        Consolidated syntheses from all layers:
        {consolidated_syntheses}

        Your tasks:
        1. Carefully consider the original prompt and all synthesis responses from each layer.
        2. Perform a thorough cross-check of all previous reasoning against the original prompt.
        3. Identify any discrepancies or logical inconsistencies between the syntheses and the original prompt.
        4. Resolve any conflicts or ambiguities, explaining your reasoning clearly.
        5. Provide a final, definitive answer to the original prompt using your own logic and reasoning.
        6. Use the syntheses as context, but do not be overly influenced by them if they contain errors.
        7. If you disagree with the conclusions in the syntheses, explain why and provide your own rationale.
        8. If you're unsure, state so and provide multiple possible answers with explanations.
        9. Aim for a comprehensive, well-reasoned response that addresses all aspects of the original prompt.
        10. Ensure your final answer is consistent with the original prompt and mathematically/logically sound.

        Structure your response clearly, addressing each of these points separately."""

        final_response = await self.final_agent.generate(final_prompt)
        return final_response

    def calculate_utilization(self, all_responses: List[List[str]], final_response: str) -> Dict[str, float]:
        utilization = {agent.name: 0 for agent in self.agents}
        total_similarity = 0

        for agent_index, agent in enumerate(self.agents):
            agent_responses = []
            for layer_responses in all_responses:
                if agent_index < len(layer_responses):
                    response_parts = layer_responses[agent_index].split(": ", 1)
                    if len(response_parts) > 1:
                        original_response = response_parts[1]
                    else:
                        original_response = response_parts[0]  # In case there's no prefix
                    agent_responses.append(original_response)
                else:
                    agent_responses.append("")  # Add empty string if response is missing
            
            combined_response = " ".join(agent_responses)
            similarity = SequenceMatcher(None, combined_response, final_response).ratio()
            utilization[agent.name] = similarity
            total_similarity += similarity

        if total_similarity == 0:
            # If total similarity is zero, assign equal utilization to all agents
            equal_share = 1.0 / len(self.agents)
            return {agent.name: equal_share * 100 for agent in self.agents}
        else:
            # Convert similarities to percentages
            return {agent_name: (similarity / total_similarity) * 100 for agent_name, similarity in utilization.items()}

def generate_markdown_report(prompt: str, all_layer_prompts: List[str], all_responses: List[List[str]], all_syntheses: List[tuple], final_response: str, utilization: Dict[str, float], agents: List[Agent], synthesis_agent: Agent, final_agent: Agent) -> str:
    markdown_content = f"""# MoA Response Report

## Original Prompt
> {prompt}

## Agent Utilization
{chr(10).join([f"- {agent_name}: {percentage:.2f}%" for agent_name, percentage in utilization.items()])}

## Final MoA Response
**Final Response Agent: {final_agent.name}**

{final_response}

---

## Intermediate Outputs
"""

    for layer, (layer_prompt, responses, (synthesis, devils_advocate)) in enumerate(zip(all_layer_prompts, all_responses, all_syntheses), 1):
        markdown_content += f"""
### Layer {layer}

<details>
<summary>Layer {layer} Details (Click to expand)</summary>

#### Layer Prompt
> {"As per original prompt" if layer == 1 else "Provided original user prompt + synthesis from previous layer"}

#### Step 1 - Agents Initial Responses
"""
        initial_responses = responses[:len(agents)] if layer == 1 else responses
        for i, (agent, response) in enumerate(zip(agents, initial_responses)):
            markdown_content += f"""
##### {agent.name} (Agent {i+1})
{response}
"""

        if layer == 1:
            markdown_content += """
#### Step 2 - Agent Aggregation of All Responses
"""
            aggregations = responses[len(agents):]
            for i, (agent, response) in enumerate(zip(agents, aggregations)):
                markdown_content += f"""
##### {agent.name} (Agent {i+1})
{response}
"""

        markdown_content += f"""
#### Step 3 - Synthesized Aggregated Responses (Synthesis Agent: {synthesis_agent.name})

##### Synthesis
{synthesis}

##### Devil's Advocate
{devils_advocate}

</details>

---
"""

    markdown_content += f"""
## Information Passed to Final Response Agent

The following synthesized information from all layers, along with the original user prompt, was passed to the final response agent ({final_agent.name}). The final agent will use this information to generate the final MoA response.

"""
    for layer, (synthesis, devils_advocate) in enumerate(all_syntheses, 1):
        markdown_content += f"""
### Layer {layer} Synthesis

{synthesis}

### Layer {layer} Devil's Advocate

{devils_advocate}

---
"""

    return markdown_content.strip()

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
    # Remove invalid filename characters and replace spaces with underscores
    sanitized = re.sub(r'[^\w\-_\. ]', '', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    # Ensure the filename is not empty and doesn't start with a dot
    if not sanitized or sanitized.startswith('.'):
        sanitized = 'untitled_' + sanitized
    return sanitized[:50]  # Limit filename length

async def interactive_session(moa: MixtureOfAgents):
    welcome_message = """
    Welcome to the Mixture of Agents (MoA) interactive session!

    Mixture of Agents is an innovative approach to leveraging multiple Large Language Models (LLMs) to enhance reasoning and language generation capabilities. This implementation is based on the paper "Mixture-of-Agents Enhances Large Language Model Capabilities" by Wang et al. (2024). You can find the paper at: https://arxiv.org/pdf/2406.04692

    In this session, we use the following models:
    1. GPT-4o
    2. Claude 3.5
    3. Gemini Pro 1.5

    How Mixture of Agents works:
    1. Each layer involves all three agents responding to the prompt or previous layer's output.
    2. After initial responses, each agent reviews and aggregates all responses, including their own.
    3. A single agent then synthesizes the aggregated responses for each layer.
    4. This process repeats for the specified number of layers.
    5. Finally, a last agent generates the final output based on all layer syntheses.

    Additional roles:
    - Synthesis Agent: Gemini Pro 1.5 is responsible for synthesizing the aggregated responses at the end of each layer.
    - Final Response Agent: Claude 3.5 is responsible for generating the final, comprehensive output.

    """
    print(welcome_message)

    print("The default is suggested as 2 layers. It is not recommended to use more than 3 layers.")
    
    layers_input = input("How many layers would you like to use? (Enter a number, or press Enter for default): ").strip()
    
    if not layers_input:
        moa.num_layers = 2
    else:
        try:
            num_layers = int(layers_input)
            if num_layers < 1:
                print("Invalid input. Using the default of 2 layers.")
                moa.num_layers = 2
            elif num_layers > 3:
                print("More than 3 layers is not recommended. Using 3 layers.")
                moa.num_layers = 3
            else:
                moa.num_layers = num_layers
        except ValueError:
            print("Invalid input. Using the default of 2 layers.")
            moa.num_layers = 2
    
    print(f"\nUsing {moa.num_layers} layers for this session.")
    print("Type your questions or prompts, and the MoA model will respond.")
    print("To exit, type 'quit', 'exit', or 'q'.")

    # Create a 'reports' folder if it doesn't exist
    reports_folder = Path("reports")
    reports_folder.mkdir(exist_ok=True)

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
        
        # Saving process
        save_output = input("\nDo you want to save the outputs? (Y/N): ").strip().lower()
        if save_output in ['y', 'yes']:
            base_filename = sanitize_filename(user_input[:50])
            
            md_filename = reports_folder / f"{base_filename}_final_response.md"
            md_content = generate_final_response_markdown(user_input, final_response, utilization)
            try:
                with open(md_filename, "w", encoding="utf-8") as f:
                    f.write(md_content)
                print(f"Final response saved as '{md_filename}'")
            except Exception as e:
                print(f"Error saving Markdown file: {str(e)}")
            
            save_full_log = input("Do you want to save full logs of the agent responses at each layer? (Y/N): ").strip().lower()
            if save_full_log in ['y', 'yes']:
                md_log_filename = reports_folder / f"{base_filename}_detailed_log.md"
                md_content = generate_markdown_report(
                    user_input,
                    all_layer_prompts,
                    all_responses,
                    all_aggregates,
                    final_response,
                    utilization,
                    moa.agents,
                    moa.synthesis_agent,
                    moa.final_agent
                )
                try:
                    with open(md_log_filename, "w", encoding="utf-8") as f:
                        f.write(md_content)
                    print(f"Detailed Markdown log saved as '{md_log_filename}'")
                except Exception as e:
                    print(f"Error saving detailed Markdown log: {str(e)}")
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
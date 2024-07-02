# Mixture of Agents (MoA) Implementation

![Image](https://github.com/AI-MickyJ/Mixture-of-Agents/assets/171421294/7064d5ea-e3dc-4baa-b399-aa30fd433f2d)

## Introduction
This project implements a Mixture of Agents (MoA) model, a novel approach to leveraging multiple Large Language Models (LLMs) to enhance reasoning and language generation capabilities. The implementation is based on the paper "Mixture-of-Agents Enhances Large Language Model Capabilities" by Wang et al. (2024). It differs in that it implements MoA with Gemini Pro 1.5, GPT4-o & Claude Sonnet 3.5 rather than open source models.

## How It Works

Our MoA implementation utilizes a multi-layer architecture with multiple LLM agents in each layer. The current setup includes:

- Three types of LLM agents:
    - GPT-4 (OpenAI)
    - Claude 3.5 Sonnet (Anthropic)
    - Gemini Pro 1.5 (Google)
- Multiple processing layers (configurable, default is 2, maximum recommended is 3)
- Specialized roles for synthesis and final output generation

## Architecture Diagram

```mermaid
graph TD
    A[User Input] --> B[Layer 1]
    B --> C[GPT-4 Agent]
    B --> D[Claude Agent]
    B --> E[Gemini Agent]
    C --> F[Aggregation & Peer Review]
    D --> F
    E --> F
    F --> G[Gemini Synthesis]
    G --> H[Layer 2]
    H --> I[GPT-4 Agent]
    H --> J[Claude Agent]
    H --> K[Gemini Agent]
    I --> L[Aggregation & Peer Review]
    J --> L
    K --> L
    L --> M[Gemini Synthesis]
    M --> N[Claude Final Output]
    N --> O[Final Response]
```

## Process Flow:

1. The user input is fed into the first layer.
2. In each layer:
   - All agents process the input simultaneously.
   - Each agent then reviews and aggregates all responses, including their own, with enhanced critical analysis.
   - Gemini synthesizes the aggregated responses and provides a devil's advocate perspective.
3. The synthesized output becomes the input for the next layer.
4. This process repeats through all layers.
5. Claude generates the final output based on all layer syntheses, performing a thorough cross-check against the original prompt.

## New Features

1. **Enhanced Aggregation**: Each agent now performs a more rigorous analysis, including assumption challenging, mathematical verification, and peer review.
2. **Devil's Advocate**: The synthesis step now includes an aggressive devil's advocate perspective to challenge prevailing answers.
3. **Logic Tree Exploration**: Agents are instructed to explore multiple interpretations using logic trees.
4. **Final Cross-Check**: The final output generation includes a thorough cross-check against the original prompt.
5. **Detailed Markdown Logging**: The system now generates comprehensive markdown logs of the entire process.

## Key Differences from the Original Paper

1. **Specialized Roles**: We use Gemini specifically for synthesis and Claude for final output, leveraging their unique strengths.
2. **Enhanced Critical Analysis**: Our implementation includes more rigorous peer review and assumption challenging at each stage.
3. **Devil's Advocate Perspective**: We've added a dedicated step to critically challenge the prevailing answers.
4. **Flexible Layer Configuration**: Users can choose the number of layers, with recommendations for optimal performance.
5. **Comprehensive Logging**: Our system provides detailed, structured logs of the entire reasoning process.

## Features

**Color-Coded CLI Output**
The CLI displays color-coded outputs for each stage of the process, enhancing readability and understanding of the workflow.

**Full Text Display**
The CLI shows the full text of each agent's response at each layer, providing a comprehensive view of the reasoning process.

**Markdown Report Generation**
After each interaction, a detailed Markdown report is generated, containing:
- The original prompt
- Full responses from each agent at each layer
- Synthesis and devil's advocate perspectives
- The final response

This report is useful for in-depth analysis of the MoA process and for sharing results.

## Installation
Follow these steps to set up the project:

1. Clone the repository:
```python
git clone https://github.com/yourusername/moa-implementation.git
cd moa-implementation
```

2. Ensure you have Docker installed on your system.

3. Create a `.env` file in the project root directory with your API keys:
```markdown
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Running the Model
To run the MoA model:

1. Build the Docker image:
```bash
docker build -t moa_project
```
2. Run the Docker container:
```bash
docker run -it --env-file .env -v "$(pwd)":/app moa_project
```

This will start an interactive session where you can enter prompts and receive responses from the MoA model.

## Output

After each interaction, you will see:

1. Color-coded intermediate outputs from each agent in the CLI.
2. A final synthesized response in the CLI.
3. An HTML report (`moa_report.html`) in your current directory with detailed outputs.

## Project Structure

- `moa_model.py`: The main implementation of the Mixture of Agents model.
- `Dockerfile`: Instucutre for building the Docker image.
- `environment.yml`: Conda environment specification.
- `requirements.txt`: List of Python package dependencies.
- `.env`: (You need to create this) Contains your API keys.
- `README.md`: This file, containing project information and instructions.

## Extending the Model
The current implementation provides a basic structure for the MoA model. You can extend it by:

1. Adding more diverse LLM agents.
2. Implementing more sophisticated routing based on task type or model strengths.
3. Experimenting with different aggregation methods.
4. Adjusting the number of layers or layer compositions.
5. Implementing error handling and rate limiting for API calls.
6. Optimizing performance with more advanced parallel processing techniques.

## References
Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., & Zou, J. (2024). Mixture-of-Agents Enhances Large Language Model Capabilities. arXiv preprint arXiv:2406.04692v1.

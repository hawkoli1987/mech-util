#!/usr/bin/env python3
"""
Generate benchmark datasets for GLM-4.6 vLLM configuration experiments.
Dataset 1: ~8K token prompts, aiming for ~8K token responses
Dataset 2: ~64K token prompts, aiming for ~64K token responses
"""

import json
import os

# Sample long-form content to pad prompts
LOREM_BLOCK = """
The field of artificial intelligence has witnessed remarkable advancements in recent years, 
particularly in the domain of large language models. These sophisticated neural networks, 
trained on vast corpora of text data, have demonstrated unprecedented capabilities in 
understanding and generating human-like text. The architecture of modern transformers, 
with their attention mechanisms and multi-head self-attention layers, enables these models 
to capture complex patterns and relationships within textual data. The training process 
involves optimizing billions of parameters through gradient descent, using techniques such 
as mixed-precision training and distributed computing across multiple GPUs or TPUs.
"""

def generate_8k_dataset(num_samples=10):
    """Generate dataset with ~8K token prompts targeting ~8K token responses."""
    
    # Pad content to reach ~8K tokens (roughly 6K words = ~8K tokens)
    padding = LOREM_BLOCK * 50  # ~5K tokens of context
    
    prompts = [
        {
            "id": f"8k_{i+1}",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a comprehensive technical writer. Provide extremely detailed, thorough responses that cover all aspects of the topic. Always aim for maximum detail and completeness."
                },
                {
                    "role": "user", 
                    "content": f"""Given the following technical context about AI systems:

{padding}

Based on this context, please write a comprehensive 8000-word technical report covering:

1. A detailed executive summary of the key concepts
2. Historical background and evolution of the technology
3. Current state-of-the-art implementations
4. Technical architecture deep-dive with diagrams described in text
5. Performance benchmarks and comparisons
6. Use cases and real-world applications (at least 10 examples)
7. Challenges and limitations with proposed solutions
8. Future directions and research opportunities
9. Ethical considerations and governance frameworks
10. Detailed appendices with technical specifications

Topic focus for this report: {get_topic(i)}

Please be as comprehensive and detailed as possible, using technical terminology and providing concrete examples throughout. Target exactly 8000 words."""
                }
            ],
            "max_tokens": 8192,
            "temperature": 0.7
        }
        for i in range(num_samples)
    ]
    return prompts


def generate_64k_dataset(num_samples=10):
    """Generate dataset with ~64K token prompts targeting ~64K token responses."""
    
    # Pad content to reach ~64K tokens (roughly 48K words = ~64K tokens)
    padding = LOREM_BLOCK * 400  # ~40K tokens of context
    
    prompts = [
        {
            "id": f"64k_{i+1}",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an exhaustive academic researcher and technical documentation specialist. Your responses must be extremely comprehensive, covering every possible angle and detail. Never summarize - always expand and elaborate extensively."
                },
                {
                    "role": "user",
                    "content": f"""The following is an extensive corpus of research materials on artificial intelligence and machine learning systems:

{padding}

Based on this extensive corpus, please produce an exhaustive academic treatise of approximately 60,000 words covering:

PART I: FOUNDATIONS (15,000 words)
- Complete historical timeline from 1950s to present
- Mathematical foundations with detailed derivations
- Computational theory and complexity analysis
- Neuroscience inspirations and biological analogies

PART II: ARCHITECTURES (15,000 words)  
- Every major neural network architecture in detail
- Attention mechanisms and transformer variants
- Training methodologies and optimization techniques
- Hardware considerations and distributed systems

PART III: APPLICATIONS (15,000 words)
- Natural language processing applications
- Computer vision systems
- Robotics and autonomous systems
- Scientific computing and simulation

PART IV: ANALYSIS (15,000 words)
- Performance benchmarking methodologies
- Comparative analysis across models
- Resource efficiency considerations
- Scaling laws and emergent capabilities

Topic specialization: {get_topic(i)}

This must be an exhaustive treatment suitable for academic publication. Include extensive citations, detailed technical specifications, code examples where relevant, and comprehensive appendices. Target exactly 60,000 words minimum."""
                }
            ],
            "max_tokens": 65536,
            "temperature": 0.7
        }
        for i in range(num_samples)
    ]
    return prompts


def get_topic(index):
    """Return different topics for variety."""
    topics = [
        "Transformer Architectures and Attention Mechanisms",
        "Distributed Training Systems for Large Language Models",
        "Mixture of Experts and Sparse Activation Patterns",
        "Inference Optimization and Serving Systems",
        "Reinforcement Learning from Human Feedback",
        "Multi-Modal Foundation Models",
        "Efficient Fine-Tuning Techniques",
        "Safety and Alignment in AI Systems",
        "Hardware Acceleration for Deep Learning",
        "Prompt Engineering and In-Context Learning"
    ]
    return topics[index % len(topics)]


def save_datasets(output_dir):
    """Generate and save both datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate datasets
    dataset_8k = generate_8k_dataset(10)
    dataset_64k = generate_64k_dataset(10)
    
    # Save as JSON
    with open(os.path.join(output_dir, "dataset_8k.json"), "w") as f:
        json.dump(dataset_8k, f, indent=2)
    
    with open(os.path.join(output_dir, "dataset_64k.json"), "w") as f:
        json.dump(dataset_64k, f, indent=2)
    
    # Print stats
    print(f"Dataset 8K: {len(dataset_8k)} samples")
    print(f"  Approx prompt tokens: ~8K per sample")
    print(f"  Target response tokens: ~8K per sample")
    
    print(f"\nDataset 64K: {len(dataset_64k)} samples")
    print(f"  Approx prompt tokens: ~64K per sample")
    print(f"  Target response tokens: ~64K per sample")
    
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    output_dir = "/scratch/Projects/SPEC-SF-AISG/source_files/Mech/storage/samples/glm46_benchmark"
    save_datasets(output_dir)

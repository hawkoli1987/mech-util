#!/usr/bin/env python3
"""
Generate benchmark datasets for GLM-4.6 vLLM configuration experiments.
Dataset 1: ~8K token prompts, aiming for ~8K token responses
Dataset 2: ~64K token prompts, aiming for ~64K token responses

Each dataset contains 100 samples for comprehensive benchmarking.
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

# Extended topic list for 100 samples
TOPICS = [
    "Transformer Architectures and Attention Mechanisms",
    "Distributed Training Systems for Large Language Models",
    "Mixture of Experts and Sparse Activation Patterns",
    "Inference Optimization and Serving Systems",
    "Reinforcement Learning from Human Feedback",
    "Multi-Modal Foundation Models",
    "Efficient Fine-Tuning Techniques (LoRA, QLoRA, Adapters)",
    "Safety and Alignment in AI Systems",
    "Hardware Acceleration for Deep Learning",
    "Prompt Engineering and In-Context Learning",
    "Knowledge Distillation and Model Compression",
    "Quantization Techniques (INT8, FP8, GPTQ, AWQ)",
    "Speculative Decoding and Draft Models",
    "KV Cache Optimization Strategies",
    "Tensor Parallelism vs Pipeline Parallelism",
    "Expert Parallelism for MoE Models",
    "Context Window Extension Techniques",
    "Rotary Position Embeddings (RoPE)",
    "Flash Attention and Memory-Efficient Attention",
    "Continuous Batching and Dynamic Batching",
    "Prefix Caching and Prompt Caching",
    "CUDA Graph Optimization",
    "NCCL and Collective Communications",
    "Model Sharding and Offloading",
    "Activation Checkpointing and Recomputation",
    "Gradient Accumulation Strategies",
    "Learning Rate Scheduling Techniques",
    "Optimizer State Sharding (ZeRO)",
    "Mixed Precision Training (FP16, BF16)",
    "Automatic Mixed Precision (AMP)",
    "Data Parallelism and Model Parallelism",
    "Sequence Parallelism for Long Contexts",
    "Ring Attention for Ultra-Long Sequences",
    "Chunked Prefill and Decode Separation",
    "Token Streaming and Incremental Decoding",
    "Beam Search vs Sampling Strategies",
    "Top-k, Top-p, and Temperature Sampling",
    "Repetition Penalty and Presence Penalty",
    "Constrained Decoding and Grammar-Based Generation",
    "Structured Output Generation (JSON, XML)",
    "Function Calling and Tool Use",
    "Retrieval-Augmented Generation (RAG)",
    "Vector Databases and Embedding Models",
    "Semantic Search and Dense Retrieval",
    "Hybrid Search (BM25 + Dense)",
    "Document Chunking Strategies",
    "Multi-Hop Reasoning and Chain-of-Thought",
    "Tree-of-Thought and Graph-of-Thought",
    "Self-Consistency and Majority Voting",
    "Constitutional AI and Red Teaming",
    "Adversarial Prompting and Jailbreaks",
    "Watermarking and Model Fingerprinting",
    "Membership Inference and Privacy",
    "Differential Privacy in Training",
    "Federated Learning for LLMs",
    "Model Merging and Ensemble Methods",
    "Sparse Upcycling and Dense-to-Sparse Conversion",
    "Continual Learning and Catastrophic Forgetting",
    "Domain Adaptation and Transfer Learning",
    "Code Generation and Program Synthesis",
    "Mathematical Reasoning and Theorem Proving",
    "Scientific Discovery with LLMs",
    "Medical AI and Clinical NLP",
    "Legal AI and Contract Analysis",
    "Financial AI and Risk Assessment",
    "Autonomous Agents and Multi-Agent Systems",
    "Planning and Task Decomposition",
    "Memory Systems for Agents",
    "Tool Integration and API Calling",
    "Browser and Computer Use Agents",
    "Robotics Foundation Models",
    "Embodied AI and Physical Reasoning",
    "Vision-Language Models (VLM)",
    "Audio-Language Models",
    "Video Understanding and Generation",
    "3D Scene Understanding",
    "World Models and Simulation",
    "Synthetic Data Generation",
    "Data Curation and Quality Filtering",
    "Deduplication and Contamination Detection",
    "Tokenizer Design and Vocabulary",
    "Byte-Level vs Subword Tokenization",
    "Multilingual Models and Cross-Lingual Transfer",
    "Low-Resource Language Support",
    "Speech Recognition and Synthesis",
    "Text-to-Speech and Voice Cloning",
    "Music Generation and Audio Synthesis",
    "Image Generation (Diffusion, GAN)",
    "Video Generation and Animation",
    "3D Asset Generation",
    "Neural Radiance Fields (NeRF)",
    "Gaussian Splatting",
    "Latent Diffusion and VAE",
    "Classifier-Free Guidance",
    "ControlNet and Conditional Generation",
    "Style Transfer and Image Editing",
    "Super-Resolution and Upscaling",
    "Object Detection and Segmentation",
    "Optical Character Recognition (OCR)",
    "Document Understanding and Layout Analysis",
]


def generate_8k_dataset(num_samples=100):
    """Generate dataset with ~8K token prompts targeting ~8K token responses."""
    
    # Pad content to reach ~8K tokens (roughly 6K words = ~8K tokens)
    padding = LOREM_BLOCK * 50  # ~5K tokens of context
    
    prompts = [
        {
            "id": f"8k_{i+1}",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a comprehensive technical writer. Provide extremely detailed, thorough responses that cover all aspects of the topic. Always aim for maximum detail and completeness. Your response should be approximately 8000 words."
                },
                {
                    "role": "user", 
                    "content": f"""Given the following technical context about AI systems:

{padding}

Based on this context, please write a comprehensive 8000-word technical report covering:

1. A detailed executive summary of the key concepts (500 words)
2. Historical background and evolution of the technology (1000 words)
3. Current state-of-the-art implementations (1500 words)
4. Technical architecture deep-dive with diagrams described in text (1500 words)
5. Performance benchmarks and comparisons (1000 words)
6. Use cases and real-world applications (at least 10 examples, 1000 words)
7. Challenges and limitations with proposed solutions (500 words)
8. Future directions and research opportunities (500 words)
9. Ethical considerations and governance frameworks (250 words)
10. Detailed appendices with technical specifications (250 words)

Topic focus for this report: {get_topic(i)}

IMPORTANT: Your response MUST be approximately 8000 words. Do not summarize or abbreviate. Provide exhaustive detail on each section."""
                }
            ],
            "max_tokens": 8192,
            "temperature": 0.7
        }
        for i in range(num_samples)
    ]
    return prompts


def generate_64k_dataset(num_samples=100):
    """Generate dataset with ~64K token prompts targeting ~64K token responses."""
    
    # Pad content to reach ~64K tokens (roughly 48K words = ~64K tokens)
    padding = LOREM_BLOCK * 400  # ~40K tokens of context
    
    prompts = [
        {
            "id": f"64k_{i+1}",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an exhaustive academic researcher and technical documentation specialist. Your responses must be extremely comprehensive, covering every possible angle and detail. Never summarize - always expand and elaborate extensively. Your response should be approximately 60,000 words."
                },
                {
                    "role": "user",
                    "content": f"""The following is an extensive corpus of research materials on artificial intelligence and machine learning systems:

{padding}

Based on this extensive corpus, please produce an exhaustive academic treatise of approximately 60,000 words covering:

PART I: FOUNDATIONS (15,000 words)
- Complete historical timeline from 1950s to present (5,000 words)
- Mathematical foundations with detailed derivations (5,000 words)
- Computational theory and complexity analysis (3,000 words)
- Neuroscience inspirations and biological analogies (2,000 words)

PART II: ARCHITECTURES (15,000 words)  
- Every major neural network architecture in detail (5,000 words)
- Attention mechanisms and transformer variants (4,000 words)
- Training methodologies and optimization techniques (3,000 words)
- Hardware considerations and distributed systems (3,000 words)

PART III: APPLICATIONS (15,000 words)
- Natural language processing applications (4,000 words)
- Computer vision systems (4,000 words)
- Robotics and autonomous systems (4,000 words)
- Scientific computing and simulation (3,000 words)

PART IV: ANALYSIS (15,000 words)
- Performance benchmarking methodologies (4,000 words)
- Comparative analysis across models (4,000 words)
- Resource efficiency considerations (4,000 words)
- Scaling laws and emergent capabilities (3,000 words)

Topic specialization: {get_topic(i)}

CRITICAL REQUIREMENTS:
- This must be an exhaustive treatment suitable for academic publication
- Include extensive citations and references throughout
- Provide detailed technical specifications for all systems discussed
- Include code examples where relevant
- Provide comprehensive appendices with additional details
- Target EXACTLY 60,000 words minimum - do not abbreviate or summarize any section"""
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
    return TOPICS[index % len(TOPICS)]


def save_datasets(output_dir, num_samples=100):
    """Generate and save both datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate datasets
    dataset_8k = generate_8k_dataset(num_samples)
    dataset_64k = generate_64k_dataset(num_samples)
    
    # Save as JSON
    with open(os.path.join(output_dir, "dataset_8k.json"), "w") as f:
        json.dump(dataset_8k, f, indent=2)
    
    with open(os.path.join(output_dir, "dataset_64k.json"), "w") as f:
        json.dump(dataset_64k, f, indent=2)
    
    # Print stats
    print(f"Dataset 8K: {len(dataset_8k)} samples")
    print(f"  Approx prompt tokens: ~6K per sample")
    print(f"  Target response tokens: ~8K per sample")
    print(f"  Total sequence length: ~14K tokens")
    
    print(f"\nDataset 64K: {len(dataset_64k)} samples")
    print(f"  Approx prompt tokens: ~48K per sample")
    print(f"  Target response tokens: ~64K per sample")
    print(f"  Total sequence length: ~112K tokens")
    
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    output_dir = "/scratch/Projects/SPEC-SF-AISG/source_files/Mech/storage/samples/glm46_benchmark"
    save_datasets(output_dir, num_samples=100)

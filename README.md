# DiscoLQA: Discourse-based Legal Question Answering

## Overview
DiscoLQA is an innovative project that addresses the challenges of applying general-purpose language models to legal texts, particularly for open-domain question answering. This repository contains the source code, datasets, and documentation for DiscoLQA, which aims to enhance the comprehension and processing of legal documents by leveraging discourse structures and abstract meaning representations (AMRs).

## Background
Legal texts, such as European legislation, often exhibit unique discursive patterns that differ significantly from ordinary language. These patterns can pose challenges for standard language models trained on non-legal corpora. DiscoLQA explores the use of Elementary Discourse Units (EDUs) and AMRs to isolate and capture these unique patterns in legal texts, thereby facilitating more effective question answering without the need for training on legal documents.

## Key Contributions
1. **Investigation of Discourse Structure in Legalese**: DiscoLQA delves into the role of discourse structure in legal texts, highlighting the differences in encoding meaning compared to ordinary language.
2. **Zero-shot Legal Question Answering**: The project demonstrates the feasibility of zero-shot question answering in legal domains using pre-trained open-domain QA systems, without the need for fine-tuning on legal texts.
3. **Q4EU Dataset**: Publication of a new evaluation dataset, Q4EU, which includes over 70 questions and 200 answers across 6 different European norms.
4. **Enhanced Answer Retrieval**: DiscoLQA showcases improved performance in F1, precision, NDCG, and MRR scores by focusing on EDUs and AMRs during information retrieval.

## Installation and Usage
To install and use DiscoLQA, clone this repository and follow the setup instructions provided in the documentation. Example usage and command-line instructions are included to facilitate easy integration and experimentation.

```bash
git clone https://github.com/Francesco-Sovrano/DiscoLQA.git
cd DiscoLQA
# Follow additional installation and usage instructions
```

## Contributing
Contributions to DiscoLQA are welcome. Please read our contributing guidelines for more information on how to submit pull requests, report issues, or suggest enhancements.

## Citations
This code is free. So, if you use this code anywhere, please cite us:
```

@article{sovrano2024discolqa,
	Author = {Sovrano, Francesco and Palmirani, Monica and Sapienza, Salvatore and Pistone, Vittoria},
	Da = {2024/01/10},
	Date-Added = {2024-01-10 13:05:06 +0000},
	Date-Modified = {2024-01-10 13:05:06 +0000},
	Doi = {10.1007/s10506-023-09387-2},
	Id = {Sovrano2024},
	Isbn = {1572-8382},
	Journal = {Artificial Intelligence and Law},
	Title = {DiscoLQA: zero-shot discourse-based legal question answering on European Legislation},
	Ty = {JOUR},
	Url = {https://doi.org/10.1007/s10506-023-09387-2},
	Year = {2024},
}
```

Thank you!

## Support

For any problem or question, please contact me at `cesco.sovrano@gmail.com`
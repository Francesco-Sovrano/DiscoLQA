# DiscoLQA: Discourse-based Legal Question Answering

Welcome to the replication package for the ICSE-SEIS 2024 paper titled ["DiscoLQA: zero-shot discourse-based legal question answering on European Legislation"](https://doi.org/10.1007/s10506-023-09387-2).

DiscoLQA is an innovative project that addresses the challenges of applying general-purpose language models to legal texts, particularly for open-domain question answering. This repository contains the source code, datasets, and documentation for DiscoLQA, which aims to enhance the comprehension and processing of legal documents by leveraging discourse structures and abstract meaning representations (AMRs).

**Open Access paper available at: [https://doi.org/10.1007/s10506-023-09387-2](https://doi.org/10.1007/s10506-023-09387-2)**

## Abstract
The structures of discourse used by legal and ordinary languages share differences that foster technical issues when applying or fine-tuning general-purpose language models for open-domain question answering on legal resources. For example, longer sentences may be preferred in European laws (i.e., Brussels I bis Regulation EU 1215/2012) to reduce potential ambiguities and improve comprehensibility, distracting a language model trained on ordinary English. In this article, we investigate some mechanisms to isolate and capture the discursive patterns of legalese in order to perform zero-shot question answering, i.e., without training on legal documents. Specifically, we use pre-trained open-domain answer retrieval systems and study what happens when changing the type of information to consider for retrieval. Indeed, by selecting only the important parts of discourse (e.g., elementary units of discourse, EDU for short, or abstract representations of meaning, AMR for short), we should be able to help the answer retriever identify the elements of interest. Hence, with this paper, we publish Q4EU, a new evaluation dataset that includes more than 70 questions and 200 answers on 6 different European norms, and study what happens to a baseline system when only EDUs or AMRs are used during information retrieval. Our results show that the versions using EDUs are overall the best, leading to state-of-the-art F1, precision, NDCG and MRR scores.

## Key Contributions
1. **Investigation of Discourse Structure in Legalese**: DiscoLQA delves into the role of discourse structure in legal texts, highlighting the differences in encoding meaning compared to ordinary language.
2. **Zero-shot Legal Question Answering**: The project demonstrates the feasibility of zero-shot question answering in legal domains using pre-trained open-domain QA systems, without the need for fine-tuning on legal texts.
3. **Q4EU Dataset**: Publication of a new evaluation dataset, Q4EU, which includes over 70 questions and 200 answers across 6 different European norms.
4. **Enhanced Answer Retrieval**: DiscoLQA showcases improved performance in F1, precision, NDCG, and MRR scores by focusing on EDUs and AMRs during information retrieval.

## System Specifications

This repository is tested and recommended on:

- OS: Linux (Debian 5.10.179 or newer) and macOS (13.2.1 Ventura or newer)
- Python version: 3.7 or newer

## Installation and Usage
To install and use DiscoLQA, clone this repository and follow the setup instructions provided in the documentation. Example usage and command-line instructions are included to facilitate easy integration and experimentation.

```bash
git clone https://github.com/Francesco-Sovrano/DiscoLQA.git
cd DiscoLQA
```

Then, you have to run the [setup.sh]([Law_EU]qa_overview/setup.sh) script to install all the dependencies in a python virtual environment.

To replicate the DiscoLQA experiment, run the [run_discolqa_experiments.sh]([Law_EU]qa_overview/run_discolqa_experiments.sh) script.

To start the web interface using MiniLM with EDU+AMR+Clauses on localhost, run 
```bash
cd [Law_EU]qa_overview
server.sh 8010 clause_edu_amr minilm
```
You'll be able to access the web-server on [http://localhost:8010](http://localhost:8010).
`8010` is the port number, you can change it.
You can also change `minilm` with `tf` or `mpnet`, and change `clause_edu_amr` with `clause`, `edu`, `amr`, `edu_amr`, `clause_edu`, etc.

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
# PRoTeD
Official implementation of PRoTeD: Structure-aware Detection of Camouflaged Machine-generated Text via Perturbation and Coherence Recovery.

PRoTeD is a structure-aware framework for detecting machine-generated and LLM-polished text. Instead of relying only on static semantic or likelihood-based signals, PRoTeD explicitly probes how textual structure responds to controlled perturbation and subsequent coherence recovery.

## Overview

PRoTeD consists of three main components:

Structure-aware perturbation: adaptively selects perturbation operators and strengths according to structural characteristics of the input text.
Coherence recovery: employs a T5-based sentence-wise reconstructor to capture recovery-related responses after perturbation.
Multi-route classification: integrates original semantic features, perturbation responses, reconstruction responses, and structural complexity metrics for final detection.

The framework is evaluated on ten benchmark subsets from CHEAT and DetectRL, covering direct generation, LLM polishing, multiple domains, and multiple LLM generators.

## Requirements

The experiments are implemented in Python with PyTorch and Hugging Face Transformers.

Main dependencies include:

Python 3.9
PyTorch
transformers
scikit-learn
numpy
pandas

Install the required packages using:

pip install -r requirements.txt

# CuLEmo: Cultural Lenses on Emotion

A benchmark for evaluating culture-aware emotion prediction capabilities of Large Language Models (LLMs) across different languages and cultural contexts.

## Paper

**CuLEmo: Cultural Lenses on Emotion -- Benchmarking LLMs for Cross-Cultural Emotion Understanding**

*Tadesse Destaw Belay, Ahmed Haj Ahmed, Alvin Grissom II, Iqra Ameer, Grigori Sidorov, Olga Kolesnikova, Seid Muhie Yimam*

### Abstract

NLP research has increasingly focused on subjective tasks such as emotion analysis. However, existing emotion benchmarks suffer from two major shortcomings: (1) they largely rely on keyword-based emotion recognition, overlooking crucial cultural dimensions required for deeper emotion understanding, and (2) many are created by translating English-annotated data into other languages, leading to potentially unreliable evaluation. To address these issues, we introduce Cultural Lenses on Emotion (CuLEmo), the first benchmark designed to evaluate culture-aware emotion prediction across six languages: Amharic, Arabic, English, German, Hindi, and Spanish. CuLEmo comprises 400 crafted questions per language, each requiring nuanced cultural reasoning and understanding. We use this benchmark to evaluate several state-of-the-art LLMs on culture-aware emotion prediction and sentiment analysis tasks. Our findings reveal that (1) emotion conceptualizations vary significantly across languages and cultures, (2) LLMs performance likewise varies by language and cultural context, and (3) prompting in English with explicit country context often outperforms in-language prompts for culture-aware emotion and sentiment understanding. We hope this benchmark guides future research toward developing more culturally aligned NLP systems.

## Overview

CuLEmo is a comprehensive benchmark for evaluating how well LLMs understand and predict emotions across different cultural contexts. The project addresses key limitations in existing emotion analysis benchmarks by:

- Incorporating cultural dimensions in emotion understanding
- Using native language data rather than translations
- Supporting multiple evaluation modes (language-specific and country-specific)
- Providing a standardized framework for cross-cultural emotion analysis

## Dataset

The benchmark includes:
- 400 carefully crafted questions per language
- Native language data 
- Cultural context-aware prompts
- Ground truth annotations for emotions and sentiment

## Installation

1. Clone the repository:
```bash
git clone git@github.com:llm-for-emotion/culemo.git
cd culemo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` with your API keys:
- `OPENAI_API_KEY` for GPT-4
- `ANTHROPIC_API_KEY` for Claude
- `GOOGLE_API_KEY` for Gemini

## Usage

### Running Inference

1. **Language-specific evaluation**:
```python
# In any of the inference scripts (e.g., OpenAI_inf.py)
LANG = "English"  # or "Arabic", "Spanish", "German", "Amharic", "Hindi"
COUNTRY = None
TSV_FILE_PATH = "data/test/eng.tsv"
```

2. **Country-specific evaluation**:
```python
LANG = None
COUNTRY = "Mexico"  # or "Ethiopia", "UAE", "Germany", "India"
TSV_FILE_PATH = "data/test/spn.tsv"
```

3. Run the inference script:
```bash
python OpenAI_inf.py  # or Anthropic_inf.py, Gemini_inf.py, ollama_inf.py
```

### Evaluation

Run the evaluation script to analyze model predictions:
```bash
python eval.py
```

## Key Findings

Our evaluation reveals several important insights:

1. Emotion conceptualizations vary significantly across languages and cultures
2. LLM performance varies by language and cultural context
3. English prompts with explicit country context often outperform in-language prompts
4. Cultural alignment is crucial for accurate emotion prediction

## Citation

If you use this benchmark in your research, please cite our paper:

```bibtex
@article{culemo2024,
  title={CULEMO: Cultural Lenses on Emotion -- Benchmarking LLMs for Cross-Cultural Emotion Understanding},
  author={Belay, Tadesse Destaw and Haj Ahmed, Ahmed and Grissom II, Alvin and Ameer, Iqra and Sidorov, Grigori and Kolesnikova, Olga and Yimam, Seid Muhie},
  journal={arXiv preprint},
  year={2024}
}
```

## License

[Your chosen license]

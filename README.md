# Does Refusal Training in LLMs Generalize to the Past Tense?

**Maksym Andriushchenko (EPFL), Nicolas Flammarion (EPFL)**

**Paper:** [https://arxiv.org/abs/2407.11969](https://arxiv.org/abs/2407.11969)

<p align="center"><img src="images/paper_first_page.png" width="700" /></p>


## Getting started
To get started, install dependencies:
`pip install transformers openai`

Make sure you have the API key stored in `OPENAI_API_KEY` and `TOGETHER_API_KEY` respectively. For this, you can run:
```
export OPENAI_API_KEY=[YOUR_API_KEY_HERE]
export TOGETHER_API_KEY=[YOUR_API_KEY_HERE]
```
Similarly, some HuggingFace models, like Llama-3, require an API key, which can be set in `HF_TOKEN`.


## Run experiments
Simply run `main.py`! :-) Examples:
```bash
python main.py --target_model=gpt-3.5-turbo --n_requests=100 --n_restarts=20
python main.py --target_model=gpt-4o-2024-05-13 --n_requests=100 --n_restarts=20 
python main.py --target_model=gpt-4o-mini --n_requests=100 --n_restarts=20 
python main.py --target_model=claude-3-5-sonnet-20240620 --n_requests=100 --n_restarts=20 
python main.py --target_model=phi3 --n_requests=100 --n_restarts=20  
python main.py --target_model=gemma2-9b --n_requests=100 --n_restarts=20 
python main.py --target_model=llama3-8b --n_requests=100 --n_restarts=20 
python main.py --target_model=r2d2 --n_requests=100 --n_restarts=20  
````


## Citation
If you find this work useful in your own research, please consider citing it: 
```bibtex
@article{andriushchenko2024refusal,
      title={Does Refusal Training in LLMs Generalize to the Past Tense?}, 
      author={Andriushchenko, Maksym and Flammarion, Nicolas},
      journal={arXiv preprint arXiv:2407.11969},
      year={2024}
}
```

### License
This codebase is released under the [MIT License](LICENSE).

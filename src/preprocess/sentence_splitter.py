# python -m src.preprocess.sentence_splitter




import re
import spacy
import numpy as np
from tqdm import tqdm
from spacy.language import Language
from src.utils.gpt2_ppl import GPT2PPLCalculator,compute_llscore_ppl
from src.utils.jsonl_handler import read_jsonl, save_results

@Language.component("prevent_split_on_decimal")
def prevent_split_on_decimal(doc):
    for i, token in enumerate(doc[:-2]):
        if i>0 and (
            token.text == "." and
            token.nbor(-1).like_num and
            token.nbor(1).like_num
        ):
            doc[i + 1].is_sent_start = False
    return doc

def compute_auto_thresholds(ppls, lls, method="percentile"):
    if method == "percentile":
        ppl_threshold = np.percentile(ppls, 95)
        llscore_threshold = np.percentile(lls, 85)
    elif method == "robust":
        ppl_median = np.median(ppls)
        ppl_iqr = np.percentile(ppls, 75) - np.percentile(ppls, 25)
        ppl_threshold = ppl_median + 1.5 * ppl_iqr

        ll_median = np.median(lls)
        ll_iqr = np.percentile(lls, 75) - np.percentile(lls, 25)
        llscore_threshold = ll_median + 1.0 * ll_iqr
    else:
        ppl_threshold = np.mean(ppls) + 1.5 * np.std(ppls)
        llscore_threshold = np.mean(lls) + 0.5 * np.std(lls)

    return {
        "ppl_threshold": ppl_threshold,
        "llscore_threshold": llscore_threshold
    }

class SentenceSegmenter:
    def __init__(self,
        enable_reference_merge=True,
        enable_ppl_merge=True,
        ppl_threshold=100,
        llscore_threshold=-60,
        max_short_len=6,
        auto_threshold=False,
        threshold_strategy="percentile",
        sample_ratio=0.1,
        strip_references=True):

        self.nlp = spacy.load("en_core_web_sm")
        if "prevent_split_on_decimal" not in self.nlp.pipe_names:
            self.nlp.add_pipe("prevent_split_on_decimal", before="parser")

        self.enable_reference_merge = enable_reference_merge
        self.enable_ppl_merge = enable_ppl_merge
        self.ppl_threshold = ppl_threshold
        self.llscore_threshold = llscore_threshold
        self.default_ppl_threshold = ppl_threshold
        self.default_llscore_threshold = llscore_threshold
        self.max_short_len = max_short_len

        self.auto_threshold = auto_threshold
        self.threshold_strategy = threshold_strategy
        self.sample_ratio = sample_ratio
        self.strip_references = strip_references

        self.dynamic_thresholds = None
        self._threshold_determined = False

        # if self.enable_ppl_merge:
            # self.ppl_model = GPT2PPLCalculator()


    def segment(self, text):
        text = text.strip()
        if self.strip_references:
            text = self._remove_reference_numbers(text)
        doc = self.nlp(text)
        sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if self.enable_reference_merge:
            sents = self._merge_reference_prefix(sents)

        if self.enable_ppl_merge:
            if self.auto_threshold and not self._threshold_determined:
                self._estimate_thresholds(sents)
            sents, scores = self._merge_based_on_ppl(sents)
        else:
            scores = [None] * len(sents)

        return [(s, p) for s, p in zip(sents, scores)]

    def _remove_reference_numbers(self, text):
        return re.sub(r"\s*\[\d+(?:[-–]\d+)?\][\.,]?", "", text)

    def _estimate_thresholds(self, sentences):
        sample_size = max(5, int(len(sentences) * self.sample_ratio))
        sample = sentences[:sample_size]
        scores = [compute_llscore_ppl(s) for s in sample]
        ppls = [p for _, p in scores]
        lls = [ll for ll, _ in scores]
        thresholds = compute_auto_thresholds(ppls, lls, method=self.threshold_strategy)
        self.ppl_threshold = thresholds["ppl_threshold"]
        self.llscore_threshold = thresholds["llscore_threshold"]
        self.dynamic_thresholds = thresholds
        self._threshold_determined = True

    def get_thresholds(self):
        return {
            "ppl_threshold": self.ppl_threshold,
            "llscore_threshold": self.llscore_threshold,
            "default_ppl": self.default_ppl_threshold,
            "default_llscore": self.default_llscore_threshold,
            "auto": self.auto_threshold,
            "strategy": self.threshold_strategy
        }

    def _merge_reference_prefix(self, sentences):
        merged = []
        i = 0
        while i < len(sentences):
            curr = sentences[i]
            if re.fullmatch(r"\[\d+\][\.,]?", curr):
                if i + 1 < len(sentences):
                    merged.append(curr + " " + sentences[i + 1])
                    i += 2
                else:
                    i += 1
            else:
                merged.append(curr)
                i += 1
        return merged

    def _merge_based_on_ppl(self, sentences):
        scores = [compute_llscore_ppl(sent) for sent in sentences]
        merged = []
        merged_scores = []
        i = 0
        while i < len(sentences):
            curr = sentences[i]
            llscore, ppl = scores[i]
            merge_condition = (
                i > 0 and
                ppl > self.ppl_threshold and
                llscore > self.llscore_threshold and
                len(curr.split()) <= self.max_short_len
            )
            if merge_condition:
                combined = merged[-1] + " " + curr
                new_ll, new_ppl =  compute_llscore_ppl(combined)
                merged[-1] = combined
                merged_scores[-1] = (new_ll, new_ppl)
                i += 1
            else:
                merged.append(curr)
                merged_scores.append((llscore, ppl))
                i += 1
        return merged, merged_scores



def split_sentences(data, auto_threshold=False, threshold_strategy="percentile"):

    segmenter = SentenceSegmenter(
        enable_reference_merge=True,
        enable_ppl_merge=True,
        auto_threshold=auto_threshold,
        threshold_strategy=threshold_strategy,
        max_short_len=6
    )
    results = []
    # for item in data:
    for item in tqdm(data):
        doc_id = item.get("id")
        text = item.get("abstract", "")
        segmented = segmenter.segment(text)
        for i, (sent, (ll, ppl)) in enumerate(segmented):
            results.append({
                "id": doc_id,
                "sentence_id": f"{doc_id}_{i}",
                "sentence": sent,
                "word_count": len(sent.split()),
                "LLScore": ll,
                "PPL": ppl
            })
    if auto_threshold:
        print("当前分句器阈值信息：", segmenter.get_thresholds())
    return results
def generate_t5_training_dataset():
    for fname in ["data/raw/ieee-init.jsonl", "data/raw/ieee-chatgpt-generation.jsonl"]:
        data = read_jsonl(fname)
        results = split_sentences(data, auto_threshold=True, threshold_strategy="percentile")
        output_file = fname.replace(".jsonl", "_split.jsonl").replace("data/raw", "data/preprocess")
        save_results(results, output_file)
        print(f"共拆分出 {len(results)} 句，保存至 {output_file}")
        # print(results)
def generate_classification_dataset():
    print("generate_classification_dataset")
    for fname in [#"data/raw/ieee-init.jsonl", "data/raw/ieee-chatgpt-generation.jsonl",
                  "data/raw/ieee-chatgpt-polish.jsonl","data/raw/ieee-chatgpt-fusion.jsonl"]:
        data = read_jsonl(fname)
        print(f"正在处理数据：{fname}")
        results = split_sentences(data, auto_threshold=True, threshold_strategy="percentile")
        output_file = fname.replace(".jsonl", "_split.jsonl").replace("data/raw", "data/classification_dataset")
        save_results(results, output_file)
        print(f"共拆分出 {len(results)} 句，保存至 {output_file}")
        # print(results)

if __name__ == "__main__":
    # generate_classification_dataset()
    data=read_jsonl("data/raw/ieee-chatgpt-polish.jsonl")
    results = split_sentences(data, auto_threshold=True, threshold_strategy="percentile")
    print(f"共拆分出 {len(results)} 句")
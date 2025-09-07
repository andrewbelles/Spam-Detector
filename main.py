#!/bin/python 
# 
# main.py  Andrew Belles  Sept 7th, 2025 
# 
# Takes in .csv file in expected format, encodes text, and makes inference 
# 

import tensorflow as tf, tensorflow_hub as hub 
import pandas as pd, numpy as np, re, joblib  

class EmailFilter():

    @staticmethod 
    def split(s: str) -> tuple[str, str]:
        """
        Split "Subject: ..." from the rest of the email body.

        Input: 
            Single string email 
        Output: 
            tuple to subject and body
        """
        # Invalid input 
        if not s:
            return "", ""
        s = s.strip().replace("\r\n", "\n").replace("\r", "\n")

        if s[:8].lower() == "subject:":
            content = s[8:].lstrip()

            # Prefer splitting at the first newline (this dataset)
            if "\n" in content:
                subject, body = content.split("\n", 1)
            else:
                # Fallback: two+ spaces delimiter (your original dataset)
                m = re.match(r"^(.*?)[ \t]{2,}(.*)$", content, flags=re.S)
                if m:
                    subject, body = m.group(1), m.group(2)
                else:
                    subject, body = content, ""

            return subject.strip(), body.strip()

        # Fallback on malformed email 
        first, _, rest = s.partition("\n")
        return first.strip(), rest.strip()


    def __init__(self, path: str):
        '''
        Instantiates encoded data from the given path for the email filter to make inferences on
        '''
        self.gen = np.random.default_rng()

        # Read in raw text 
        raw = pd.read_csv(path, header=0, dtype={"text": str, "spam": int},
                          na_filter=False, quotechar='"', escapechar='\\')
        # Split text by subject and body 
        split_emails = raw["text"].apply(EmailFilter.split)
        sb = pd.DataFrame(split_emails.tolist())
        sb.columns = ["subject", "body"]

        # Get model text to be encoded 
        sb["model_text"] = ("[SUBJECT] " + sb["subject"] + 
                            " [BODY] " + sb["body"])
        # Create dataframe using model text and labels 
        self.emails = pd.DataFrame({"text": sb["model_text"], "y": raw["spam"].values})
    
        self.USE4 = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                   trainable=False)
        
        # Encode data numerically for inference 
        self.Subjects = sb["subject"]   # Get subjects to relate back to numeric data 
        self.X = self.encode_().astype(np.float64)
        self.y = self.emails["y"].to_numpy().astype(np.int32)

        # Get on record model 
        self.model = joblib.load("model.joblib")


    def encode_(self, batch_size=64):
        '''
        
        Does not mutate self

        Inputs:
            Subject, body dataframe from self 
            single batch size to process through encoder each time 
        Output: 
            Vector output of encoder for each email string. L2 normalized 
        '''
        # Get universal-sentence-encoder from tensorflow_hub
        # Process emails in batches
        vecs = []
        text = self.emails["text"].tolist()
        for i in range(0, len(text), batch_size):
            batch = tf.constant(text[i:i+batch_size])
            v = self.USE4(batch)
            v = tf.nn.l2_normalize(v, axis=1).numpy()
            vecs.append(v)
        return np.vstack(vecs)


    def infer(self) -> tuple[list[str], float]:
        # Make inference, keep all inferences above 0.5 threshold (Can be tighter bound? TODO)
        probabilities = self.model.predict_proba(self.X)[:, 1]
        preds = (probabilities >= 0.5).astype(np.int32)

        guess_tags = np.where(preds == 1, "spam", "ham")
        real_tags  = np.where(self.y == 1, "spam", "ham")

        error_rate = 1.0 - np.mean(np.ravel(preds) != np.ravel(self.y))

        # For each prediction, print the subject and whether it was spam or ham 
        for i, (tag, subject) in enumerate(zip(guess_tags, self.Subjects)):
            print(f"[SUBJECT] {subject} [GUESS] {tag} [REAL] {real_tags[i]}")

        return list(guess_tags), error_rate 

    
    def confusion(self):
        pass 


def main():

    emails1 = EmailFilter("emails.csv")
    tags, control_rate = emails1.infer()
    assert(tags is not None)
    print(f"Control Rate = {control_rate}")

    emails2 = EmailFilter("emails2.csv")
    tags, test_rate = emails2.infer()
    assert(tags is not None)
    print(f"Test Rate = {test_rate}")


if __name__ == "__main__":
    main()

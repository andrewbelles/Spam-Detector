#!/bin/bash 
# 
# text_encoding.py  Andrew Belles  Sept 6th, 2025
# Takes in spam classification data from csv, encodes using USE-4 
# Normalized data, splits into training and test samples, and saves to file 
# 

import tensorflow as tf, tensorflow_hub as hub 
import pandas as pd, numpy as np, re 
from sklearn.model_selection import train_test_split 

# CONST pattern to capture body after subject and split at 2+ spaces  
SB_RE = re.compile(r"^\s*Subject:\s*(.+?)(?:\s{2,}|$)(.*)$", flags=re.I|re.S)

# Load universal sentence encoder 
USE = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                     trainable=False)


def load_samples(path: str):
    '''
    Load X and y from requested the .npz file  
    Input: 
        path to .npz 
    Outputs: 
        tuple to X, y
    '''
    data = np.load(path, mmap_mode="r")
    X = data["X"]
    y = data["y"]
    data.close()

    y = np.asarray(y).reshape(-1).astype(np.int32)
    X = np.asarray(X).astype(np.float32, copy=False)

    assert X.shape[0] == y.shape[0]
    return X, y 


def split_subject_body(s: str) -> tuple[str, str]:
    '''
    input: str 
    rtype: tuple[str, str]
    
    Converts single row of email into subject and body of email 
    '''
    m = SB_RE.match(s.strip())
    if m:
        subject, body = m.group(1).strip(), m.group(2).strip()
        return subject, body 
    # If fails to match (Shouldn't be an issue)
    return "", s.strip()


def encode_emails(emails, batch_size=64): 
    '''
    Inputs:
        list of [SUBJECT] ... [BODY] ... email strings
        single batch size to process through encoder each time 
    Output: 
        Vector output of encoder for each email string. L2 normalized 
    '''
    vecs = []
    for i in range(0, len(emails), batch_size):
        batch = tf.constant(emails[i:i+batch_size])
        v = USE(batch)
        v = tf.nn.l2_normalize(v, axis=1).numpy()
        vecs.append(v)
    return np.vstack(vecs)


def main():
    '''
    Purpose: 
        Cleans emails, passes through USE-4 encoder in batches, 
        Reconstructs DataFrame with encoded vectors and label,
    
    Output: 
        Saves final DataFrame to email_dataset.csv
    '''
    raw = pd.read_csv("emails.csv", 
                      header=0, 
                      dtype={"text": str, "spam": int}, 
                      na_filter=False, 
                      quotechar='"',
                      escapechar='\\')
    
    # Split subject from body and reconstruct DataFrame 
    split_emails = raw["text"].apply(split_subject_body)
    subj_body    = pd.DataFrame(split_emails.tolist())
    subj_body.columns = ["subject", "body"]

    subj_body["model_text"] = ("[SUBJECT] " + subj_body["subject"] + 
                               " [BODY] " + subj_body["body"])
    gen = np.random.default_rng()
    emails = subj_body.sample(frac=1, random_state=gen).reset_index(drop=True)
   
    # Get encoded emails from USE-4 encoder 
    X = encode_emails(emails["model_text"])
    y = raw["spam"].to_numpy().astype(int)

    # Into numpy datatypes
    X = X.astype(np.float64)
    y = y.astype(np.int32).reshape(-1, 1)
    
    # Get indices for training and test samples, while maintaining class balance 
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, 
                                           test_size=0.20, 
                                           stratify=y, 
                                           random_state=0, 
                                           shuffle=True)
    
    # Pull out each dataset
    X_train, y_train = X[idx_train], y[idx_train]
    X_test,  y_test  = X[idx_test], y[idx_test]

    # Save to datafiles 
    np.savez_compressed("training_samples.npz", 
                        X=X_train.astype(np.float64),
                        y=y_train.astype(np.int32),
                        idx=idx_train)
    np.savez_compressed("test_samples.npz", 
                        X=X_test.astype(np.float64),
                        y=y_test.astype(np.int32),
                        idx=idx_test)


if __name__ == "__main__":
    main()

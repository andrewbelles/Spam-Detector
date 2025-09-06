#import tensorflow as tf, tensorflow_hub as hub 
import pandas as pd, numpy as np, re 

# CONST pattern to capture body after subject and split at 2+ spaces  
SB_RE = re.compile(r"^\s*Subject:\s*(.+?)(?:\s{2,}|$)(.*)$", flags=re.I|re.S)

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

def main():
    # Load universal sentence encoder 
    #use = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
    #                     trainable=False)
    # TODO: Fix label context being lost 
    raw = pd.read_csv("emails.csv", 
                      header=0, 
                      dtype={"text": str, "spam": int}, 
                      na_filter=False, 
                      quotechar='"',
                      escapechar='\\')
    
    # Split subject from body and reconstruct DataFrame 
    emails = raw["text"].apply(split_subject_body)
    output = pd.DataFrame(emails.tolist())
    output.columns = ["subject", "body"]
    output["spam"] = raw["spam"].astype(int)

    # Randomize ordering 
    gen = np.random.default_rng()
    output = output.sample(frac=1, random_state=gen).reset_index(drop=True)
    output.to_csv("cleaned_emails.csv", index=False)

main()

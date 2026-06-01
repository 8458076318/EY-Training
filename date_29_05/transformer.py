from transformers import pipeline

# TODO 1: Initialise a sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Sample sentences relevant to consulting work
sentences = [
    "The client was very satisfied with the delivery.",
    "The project is significantly over budget and behind schedule.",
    "The new regulatory framework presents both risks and opportunities.",
    "The team delivered exceptional results despite the tight deadline.",  # TODO 2: added sentence
]

# TODO 3: Run the classifier on all sentences and print results
# Expected output format:
# "The client was satisfied..." → POSITIVE (0.9987)
for sentence in sentences:
    result = getSentiment(classifier)
    label = result[0]['label']
    score = result[0]['score']
    print(f'"{sentence[:50]}..." → {label} ({score:.4f})')

def getSentiment(classifier):
    return classifier(sentence)
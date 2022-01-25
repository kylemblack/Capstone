from google.cloud import vision
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import io

def cos_sim(string1, string2):
    """Returns cosine similarity of two strings as a percentage"""
    string1token = word_tokenize(string1)
    string2token = word_tokenize(string2)

    string1token = {w for w in string1token if not w in stopwords.words('english')}
    string2token = {w for w in string2token if not w in stopwords.words('english')}

    #string1token = {w for w in string1token}
    #string2token = {w for w in string2token}

    vector1 = []
    vector2 = []

    combined_vector = string1token.union(string2token)

    for w in combined_vector:
        if w in string1token:
            vector1.append(1)
        else:
            vector1.append(0)
        if w in string2token:
            vector2.append(1)
        else:
            vector2.append(0)

    c = 0

    for i in range(len(combined_vector)):
        c+= vector1[i]*vector2[i]
    
    cosine = c / float((sum(vector1) * sum(vector2))**0.5)

    return cosine

def print_text(path):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    full_text = ''

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    full_text += word_text + ' '

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return full_text

if __name__ == "__main__":
    image_location1 = input("Enter first image path: ")
    image_location2 = input("Enter second image path: ")

    print("Reading Image 1.")
    text1 = print_text(image_location1)
    print("Reading Image 2.")
    text2 = print_text(image_location2)

    print("Comparing")
    percent_sim = cos_sim(text1, text2) * 100
    percent_sim = round(percent_sim, 1)

    print(text1)
    print("")
    print(text2)
    print("")
    print("The documents are %{} similar.".format(percent_sim))
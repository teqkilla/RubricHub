import collections
import json
import random
import re
import string
from typing import Dict, Optional, Sequence, Union, List, Any

import nltk
custom_path = "your/nltk_data/path" 
nltk.data.path.append(custom_path)

print("Current NLTK data paths:")
for path in nltk.data.path:
    print(f"  - {path}")


import langdetect
from absl import logging

import instructions_util


def _filter_think_tags( text: str) -> str:
        """Run the `_filter_think_tags` verification."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _verify_thinking_format(model_response: str, parameters: Dict[str, Any])  -> bool:
    """Run the `_verify_thinking_format` verification."""
    open_tag = "<think>\n"
    close_tag = "\n</think>"

    
    if not model_response.startswith(open_tag):
        return True

    open_cnt = model_response.count(open_tag)
    close_cnt = model_response.count(close_tag)
    if open_cnt != 1 or close_cnt != 1:
        return True

    start = model_response.find(open_tag)
    end = model_response.find(close_tag)

    
    if end < start + len(open_tag):
        return True
    return False



def math_verify(model_response: str, parameters: Dict[str, Any]) -> bool:
    """
    Verify model response using mathematical verification logic
    Priority: math_verify.py first, then fallback to dapo verification if failed
    
    Args:
        model_response: Model's response
        parameters: Verification parameter dictionary, format: {"answer": "answer"}
        
    Returns:
        bool: Verification result, True indicates correct, False indicates incorrect
    """
    
    from verl.utils.reward_score.math_verify import compute_score as compute_score_verify
    from verl.utils.reward_score.math_dapo import compute_score as compute_score_dapo
    
    # Extract answer from dictionary
    answer = parameters.get("answer")
    if answer is None:
        raise ValueError("math_verify: parameters dict must contain 'answer' key")
    
    # Priority 1: Use math_verify.py verification
    math_verify_passed = False
    try:
        score_verify = compute_score_verify(model_response, answer)
        if score_verify > 0.5:
            math_verify_passed = True
            return True
    except Exception as e:
        print(f"math_verify: Verify verification failed with error: {e}")
    
    # Priority 2: Fallback to dapo verification if verify verification failed
    try:
        result_dapo = compute_score_dapo(model_response, answer, strict_box_verify=True)
        acc_dapo = result_dapo.get("acc", False)
        
        # If math_verify failed but dapo passed, print the response and answer
        if not math_verify_passed and acc_dapo:
            print(f"math_verify: math_verify failed but dapo passed")
            print(f"  Response: {model_response}")
            print(f"  Answer: {answer}")
        
        return acc_dapo
    except Exception as e:
        print(f"math_dapo: Dapo verification failed with error: {e}")
        return False


def word_count_range(model_response: str, parameters: Dict[str, Any]) -> bool:
    """Run the `word_count_range` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.

    
    min_count = parameters.get("min_count")
    max_count = parameters.get("max_count")
    
    if min_count is None or max_count is None:
        raise ValueError("word_count_range: parameters dict must contain 'min_count' and 'max_count' keys")
    
    try:
        min_count = int(min_count)
        max_count = int(max_count)
    except (ValueError, TypeError):
        raise ValueError(f"word_count_range: min_count and max_count must be integers, got min_count={min_count}, max_count={max_count}")
    
    if min_count > max_count:
        raise ValueError(f"word_count_range: min_count ({min_count}) cannot be greater than max_count ({max_count})")
        
    word_count = len(model_response.split())
    
    result = min_count <= word_count <= max_count
    #print(f"word_count_range: response has {word_count} words, range [{min_count}, {max_count}], result: {result}")
    return result



# --- ifeval G ---

_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = instructions_util.LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("less than", "at least")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = ("My answer is yes.", "My answer is no.", "My answer is maybe.")

# The options of starter keywords.
_STARTER_OPTIONS = (
    "I would say",
    "My answer is",
    "I believe",
    "In my opinion",
    "I think",
    "I reckon",
    "I feel",
    "From my perspective",
    "As I see it",
    "According to me",
    "As far as I'm concerned",
    "To my understanding",
    "In my view",
    "My take on it is",
    "As per my perception",
)

# The options of ending keywords.
# TODO(jeffreyzhou) add more ending options
_ENDING_OPTIONS = ("Any other questions?", "Is there anything else I can help with?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
_SECTION_SPLITER = ("Section", "SECTION")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500

# phrases
_PHRASES = [
    "Dance like nobody is watching you",
    "The early bird catches the worm",
    "Time flies when having fun",
    "Every cloud has a silver lining",
    "Actions speak louder than words",
    "Don't judge a book by cover",
    "Live each day to the fullest",
    "All that glitters is not gold",
    "Laughter is the best medicine",
    "The pen is mightier than sword",
]




### 1. ResponseLanguageChecker
def ResponseLanguageChecker(model_response: str, parameters: dict) -> bool:
    """
    Check if the entire response is in the expected language.

    Args:
        model_response (str): The response string to check.
        parameters (dict): 
            - 'language': str, A dictionary with key "language", whose value is the expected ISO 639-1 code.
 
    Returns:
        bool: True if the detected language matches the expected language, False otherwise.
    """
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    assert isinstance(model_response, str)

    language = parameters.get("language")
    if language is None:
        # language = random.choice(list(_LANGUAGES.keys()))
        raise ValueError(f"language is None, please check!")

    try:
        return langdetect.detect(model_response) == language
    except langdetect.LangDetectException as e:
        
        logging.error("Unable to detect language for text %s due to %s", model_response, e)
        return True

### 2.NumberOfSentences
def NumberOfSentences(model_response: str, parameters: dict) -> bool:
    """
    Check if the number of sentences in the response satisfies the condition.

    Args:
        model_response: The response string to check.
        parameters: A dictionary with keys:
            - "num_sentences": int, the threshold number of sentences.
            - "relation": str, either "less than" or "at least".

    Returns:
        True if the response follows the instruction, False otherwise.
    """
    model_response = _filter_think_tags(model_response) # Remove the thinking block.

    
    num_sentences_threshold = parameters.get("num_sentences")
    relation = parameters.get("relation")

    
    if num_sentences_threshold is None or num_sentences_threshold < 0:
        num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

    if relation is None:
        # relation = random.choice(_COMPARISON_RELATION)
        raise ValueError(f"relation is None, please check!")
    elif relation not in _COMPARISON_RELATION:
        raise ValueError(
            f"The supported relation for comparison must be in {_COMPARISON_RELATION}, "
            f"but {relation} is given."
        )
    
    num_sentences = instructions_util.count_sentences(model_response)


    
    if relation == "less than":
        return num_sentences < num_sentences_threshold
    elif relation == "at least":
        return num_sentences >= num_sentences_threshold
    else:
        raise ValueError("Unsupported relation type.")


### 3.PlaceholderChecker
def PlaceholderChecker(model_response: str, parameters: dict) -> bool:
    """Check the placeholders in template writing.

    Args:
      model_response: A string representing the model's response.
      parameters: 
      - num_placeholders: int, A dictionary containing 'num_placeholders' specifying the  minimum number of placeholders required.

    Returns:
      True if the actual number of placeholders in the response is greater than
      or equal to `num_placeholders`; otherwise, False.
    """
    model_response = _filter_think_tags(model_response) # Remove the thinking block.

    num_placeholders = parameters.get("num_placeholders", None)
    
    # If num_placeholders is None or negative, assign a random value
    if num_placeholders is None or num_placeholders < 0:
        # num_placeholders = random.randint(1, 5)  # Replace 5 with an appropriate constant if needed
        raise ValueError(f"num_placeholders is None, please check!")
    
    # Find placeholders in the response
    placeholders = re.findall(r"\[.*?\]", model_response)
    num_found_placeholders = len(placeholders)
    
    # Check if the number of placeholders is sufficient
    return num_found_placeholders >= num_placeholders


### 4.BulletListChecker
def BulletListChecker(model_response: str, parameters: dict) -> bool:
    """Checks the bullet list in the prompt.
        parameters:
        - num_bullets: int.
    """
    model_response = _filter_think_tags(model_response) # Remove the thinking block.

    num_bullets = parameters.get('num_bullets', None)
    
    if num_bullets is None or num_bullets < 0:
        # num_bullets = random.randint(1, _NUM_BULLETS)
        raise ValueError(f"num_bullets is None, please check!")

    
    description_pattern = (
        f"Your answer must contain exactly {num_bullets} bullet points. "
        + "Use the markdown bullet points such as:\n"
        + "* This is point 1. \n"
        + "* This is point 2"
    )

    bullet_lists = re.findall(r"^\s*\*[^\*].*$", model_response, flags=re.MULTILINE)
    bullet_lists_2 = re.findall(r"^\s*-.*$", model_response, flags=re.MULTILINE)
    num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
    
    return num_bullet_lists == num_bullets

# 5.ConstrainedResponseChecker
def ConstrainedResponseChecker(model_response: str, parameters: dict) -> bool:
    """Run the `ConstrainedResponseChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.

    value = model_response.strip()
    
    for constrained_response in _CONSTRAINED_RESPONSE_OPTIONS:
        if constrained_response in value:
            return True
    return False



# 6.ConstrainedStartChecker
def ConstrainedStartChecker(model_response: str, parameters: dict) -> bool:
    """Checks if the response starts with the constrained keyword or phrase."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.

    starter = parameters.get('starter', None)
    if isinstance(starter, str):
        starter = starter.strip() if starter else None
    if starter is None:
        # starter = random.choice(_STARTER_OPTIONS)
        raise ValueError(f"starter is None, please check!")

    # Check if the response starts with the constrained starter keyword or phrase
    response_pattern = r"^\s*" + re.escape(starter) + r".*$"
    response_with_constrained_start = re.search(response_pattern, model_response, flags=re.MULTILINE)
    return bool(response_with_constrained_start)

# 7.HighlightSectionChecker
def HighlightSectionChecker(model_response: str, parameters: dict) -> bool:
    """Checks if the number of highlighted sections meets the requirement.

    Args:
      model_response: A string representing the response. The response is expected to
        contain highlighted sections in the format of *highlighted* or **highlighted**.

      parameters: A dictionary containing the parameter 'num_highlights', which specifies
        the minimum number of highlighted sections required.

    Returns:
      True if the actual number of highlighted sections in the format of *highlighted sections*
      meets the minimum requirement; otherwise False.
    """
    model_response = _filter_think_tags(model_response) # Remove the thinking block.

    num_highlights = parameters.get('num_highlights', 0)
    if num_highlights < 0:
        # num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)
        raise ValueError(f"num_highlights is None, please check!")

    num_highlights_found = 0
    highlights = re.findall(r"\*[^\n\*]*\*", model_response)
    double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", model_response)

    for highlight in highlights:
        if highlight.strip("*").strip():
            num_highlights_found += 1
    for highlight in double_highlights:
        if highlight.removeprefix("**").removesuffix("**").strip():
            num_highlights_found += 1

    return num_highlights_found >= num_highlights



# 8.HighlightSectionChecker
def SectionChecker(model_response: str, parameters: dict) -> bool:
    """Checks the sections in the response.

    Args:
      model_response: A string representing the response to check.
      parameters: A dictionary containing 'section_spliter' and 'num_sections'.
        - 'section_spliter' (str): The section split keyword. ['Section', 'SECTION']
        - 'num_sections' (int): The minimum number of sections.
    
    Example
    _SECTION_SPLITER = ['Section', 'SECTION']
    _NUM_SECTIONS = 5  # You can adjust this based on your needs.

    Returns:
      bool: True if the response contains the required number of sections, False otherwise.
    """
    model_response = _filter_think_tags(model_response) # Remove the thinking block.

    section_spliter = parameters.get("section_spliter", None)
    num_sections = parameters.get("num_sections", None)

    if section_spliter is None:
        raise ValueError(f"section_spliter is None, please check!")

    if num_sections is None:
        raise ValueError(f"num_sections is None, please check!")

    section_splitter_patten = r"\s?" + section_spliter + r"\s?\d+\s?"
    sections = re.split(section_splitter_patten, model_response)
    num_found_sections = len(sections) - 1

    return num_found_sections >= num_sections


# 9.ParagraphChecker
def ParagraphChecker(model_response: str, parameters: dict) -> bool:
    """
    Checks if the response contains the required number of paragraphs.
    
    Args:
        model_response (str): The response text to check.
        parameters (dict): A dictionary containing the 'num_paragraphs' key to specify the required number of paragraphs.
        
    Returns:
        bool: True if the number of paragraphs matches the required count, otherwise False.
    """
    model_response = _filter_think_tags(model_response) # Remove the thinking block.

    num_paragraphs = parameters.get('num_paragraphs', None)  

    if num_paragraphs is None:
        raise ValueError(f"num_paragraphs is None, please check!")
    
    # Split the response into paragraphs using the markdown divider: ***
    paragraphs = re.split(r"\s?\*\*\*\s?", model_response)
    num_paragraphs_actual = len(paragraphs)

    # Check if paragraphs are non-empty and validate the count
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():  # If the paragraph is empty
            if index == 0 or index == len(paragraphs) - 1:
                num_paragraphs_actual -= 1
            else:
                return False

    # Check if the number of actual paragraphs matches the required number
    return num_paragraphs_actual == num_paragraphs


# 10.PostscriptChecker
def PostscriptChecker(model_response: str, parameters: dict) -> bool:
    """Run the `PostscriptChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.

    postscript_marker = parameters['postscript_marker']
    value = model_response.lower()

    if postscript_marker == "P.P.S":
        postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        postscript_pattern = r"\s*p\.\s?s\..*$"
    else:
        
        postscript_pattern = r"\s*" + postscript_marker.lower() + r".*$"

    postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
    return bool(postscript)


# 11.RephraseChecker
def RephraseChecker(model_response: str, parameters: dict) -> bool:
    """Run the `RephraseChecker` verification."""

    def is_change(response: str) -> bool:
        """Run the `is_change` verification."""
        return re.search(r"\*.*\*", response) is not None

    def strip_changes(response: str) -> str:
        """Run the `strip_changes` verification."""
        return re.sub(r"\*.*\*", "", response)

    if not is_change(model_response):
        raise ValueError(f"Model response '{model_response}' does not contain an asterisk-marked span to rewrite.")

    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    original_message = parameters['original_message']

    response_without_changes = strip_changes(model_response)
    reference_without_changes = strip_changes(original_message)

    return response_without_changes == reference_without_changes

# 12.KeywordChecker
def KeywordChecker(model_response: str, parameters: dict) -> bool:
    """Check the existence of certain keywords in the response.

    Args:
        model_response: A string representing the response to be checked.
        parameters: A dictionary containing the 'keywords' key, which holds 
                    the expected list of keywords.

    Returns:
        bool: True if all keywords are present in the response, False otherwise.
    """
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    keywords = parameters.get('keywords', [])
    
    if not keywords:
        # keywords = instructions_util.generate_keywords(num_keywords=_NUM_KEYWORDS)
        raise ValueError(f"keywordss is None, please check!")

    
    keywords = sorted(keywords)
    
    # Check if the model response contains all the keywords
    for keyword in keywords:
        if not re.search(keyword, model_response, flags=re.IGNORECASE):
            return False
    return True


# 12.KeywordFrequencyChecker
def KeywordFrequencyChecker(model_response: str, parameters: dict) -> bool:
    """Check if the keyword appears with the required frequency.

    Args:
      model_response: The response from the model to check.
      parameters: A dictionary containing 'keyword', 'frequency', and 'relation'.

    Returns:
      A boolean indicating whether the response satisfies the frequency condition.
    """
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    keyword = parameters.get("keyword")
    frequency = parameters.get("frequency")
    relation = parameters.get("relation")

    if not keyword:
        # keyword = random.choice(["example"])  # Replace with your method of generating keywords
        raise ValueError(f"keyword is None, please check!")
    else:
        keyword = keyword.strip()

    if frequency is None or frequency < 0:
        # frequency = random.randint(1, _KEYWORD_FREQUENCY)
        raise ValueError(f"frequency is None, please check!")


    if relation not in _COMPARISON_RELATION:
        raise ValueError(f"The supported relation for comparison must be in ['less than', 'at least'], but {relation} is given.")

    actual_occurrences = len(re.findall(keyword, model_response, flags=re.IGNORECASE))

    if relation == _COMPARISON_RELATION[0]:  # 'less than'
        return actual_occurrences < frequency
    elif relation == _COMPARISON_RELATION[1]:  # 'at least'
        return actual_occurrences >= frequency

def NumberOfWords(model_response: str, parameters: dict) -> bool:
    """Run the `NumberOfWords` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    num_words_target = parameters.get("num_words")
    relation = parameters.get("relation")

    if num_words_target is None or relation is None:
        raise ValueError("The 'parameters' dictionary must contain 'num_words' and 'relation' keys.")

    
    actual_num_words = instructions_util.count_words(model_response)

    if relation == 'less than':
        return actual_num_words < num_words_target
    elif relation == 'at least':
        return actual_num_words >= num_words_target
    else:
        raise ValueError(f"Unsupported relation: '{relation}'. Only 'less than' and 'at least' are supported.")

def JsonFormat(model_response: str, parameters: dict) -> bool:
    """Run the `JsonFormat` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    value = (
        model_response.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
    except ValueError:
        return False
    return True

def ParagraphFirstWordCheck(model_response: str, parameters: dict) -> bool:
    """Run the `ParagraphFirstWordCheck` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    
    required_num_paragraphs_val = parameters.get("num_paragraphs")
    nth_paragraph_index_val = parameters.get("nth_paragraph")
    required_first_word = parameters.get("first_word")

    
    if required_num_paragraphs_val is None or nth_paragraph_index_val is None or required_first_word is None:
        return False

    
    
    try:
        required_num_paragraphs = int(required_num_paragraphs_val)
        nth_paragraph_index = int(nth_paragraph_index_val)
    except (ValueError, TypeError):
        
        return False
    # -----------------------------------------

    
    paragraphs = [p.strip() for p in re.split(r"\n\n", model_response) if p.strip()]
    actual_num_paragraphs = len(paragraphs)

    
    if actual_num_paragraphs != required_num_paragraphs:
        return False

    
    
    if not (1 <= nth_paragraph_index <= actual_num_paragraphs):
        return False

    
    target_paragraph = paragraphs[nth_paragraph_index - 1]

    
    words = target_paragraph.split()
    if not words:
        
        return False

    
    first_word_raw = words[0]

    
    
    cleaned_first_word = first_word_raw.strip(string.punctuation)

    
    return cleaned_first_word.lower() == required_first_word.lower()

def ForbiddenWords(model_response: str, parameters: dict) -> bool:
    """Run the `ForbiddenWords` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    forbidden_words = parameters.get("forbidden_words", [])
    for word in forbidden_words:
        
        
        if re.search(r"\b" + re.escape(word) + r"\b", model_response, flags=re.IGNORECASE):
            return False
    return True

def RephraseParagraph(model_response: str, parameters: dict) -> bool:
    """Run the `RephraseParagraph` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    original_paragraph = parameters["original_paragraph"]
    low = parameters["low"]
    high = parameters["high"]

    
    val_words = re.findall(r"\w+", model_response.lower())
    original_words = re.findall(r"\w+", original_paragraph.lower())

    
    dict_val = collections.Counter(val_words)
    dict_original = collections.Counter(original_words)

    similar_words = 0
    
    for word, count in dict_original.items():
        
        similar_words += min(count, dict_val.get(word, 0))

    
    return low <= similar_words <= high

def TwoResponsesChecker(model_response: str, parameters: dict) -> bool:
    """Run the `TwoResponsesChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    valid_responses = []
    responses = model_response.split("******")
    for index, response in enumerate(responses):
        
        if not response.strip():
            
            if index != 0 and index != len(responses) - 1:
                return False
        else:
            
            valid_responses.append(response)
    
    return len(valid_responses) == 2 and valid_responses[0].strip() != valid_responses[1].strip()

def RepeatPromptThenAnswer(model_response: str, parameters: dict) -> bool:
    """Run the `RepeatPromptThenAnswer` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    prompt_to_repeat = parameters["prompt_to_repeat"]
    
    if model_response.strip().lower().startswith(prompt_to_repeat.strip().lower()):
        return True
    return False

def EndChecker(model_response: str, parameters: dict) -> bool:
    """Run the `EndChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    end_phrase = parameters["end_phrase"]
    
    processed_response = model_response.strip().strip('"').lower()
    
    processed_end_phrase = end_phrase.strip().lower()
    
    return processed_response.endswith(processed_end_phrase)

def TitleChecker(model_response: str, parameters: dict) -> bool:
    """Run the `TitleChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    
    pattern = r"<<[^\n]+>>"
    re_pattern = re.compile(pattern)
    titles = re.findall(re_pattern, model_response)

    for title in titles:
        
        if title.lstrip("<").rstrip(">").strip():
            return True
    return False

def LetterFrequencyChecker(model_response: str, parameters: dict) -> bool:
    """Run the `LetterFrequencyChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    letter = parameters['letter'].lower()
    frequency = parameters['let_frequency']
    relation = parameters['let_relation']

    supported_relations = ['less than', 'at least']
    if relation not in supported_relations:
        raise ValueError(
            f"The comparison relation must be one of {supported_relations}, but got {relation}."
        )

    
    response_lower = model_response.lower()
    
    letter_counts = collections.Counter(response_lower)
    actual_count = letter_counts.get(letter, 0)

    
    if relation == 'less than':
        return actual_count < frequency
    else:  # 'at least'
        return actual_count >= frequency

def CapitalLettersEnglishChecker(model_response: str, parameters: dict) -> bool:
    """Run the `CapitalLettersEnglishChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    
    if not model_response.isupper():
        return False

    try:
        
        return langdetect.detect(model_response) == "en"
    except langdetect.LangDetectException:
        
        return True

def LowercaseLettersEnglishChecker(model_response: str, parameters: dict) -> bool:
    """Run the `LowercaseLettersEnglishChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    
    if not model_response.islower():
        return False

    try:
        
        return langdetect.detect(model_response) == "en"
    except langdetect.LangDetectException:
        
        return True

def CommaChecker(model_response: str, parameters: dict) -> bool:
    """Run the `CommaChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    
    return "," not in model_response

def CapitalWordFrequencyChecker(model_response: str, parameters: dict) -> bool:
    """Run the `CapitalWordFrequencyChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    frequency = parameters['capital_frequency']
    relation = parameters['capital_relation']

    supported_relations = ['less than', 'at least']
    if relation not in supported_relations:
        raise ValueError(
            f"The comparison relation must be one of {supported_relations}, but got {relation}."
        )

    
    words = nltk.word_tokenize(model_response)
    
    
    capital_words_count = sum(1 for word in words if word.isalpha() and word.isupper())

    if relation == 'less than':
        return capital_words_count < frequency
    else:  # 'at least'
        return capital_words_count >= frequency

def QuotationChecker(model_response: str, parameters: dict) -> bool:
    """Run the `QuotationChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    
    value = model_response.strip()
    
    return len(value) > 1 and value.startswith('"') and value.endswith('"')

def RepeatPhraseChecker(model_response: str, parameters: dict) -> bool:
    """Run the `RepeatPhraseChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    
    phrase = parameters.get("phrase")
    small_n = parameters.get("small_n")

    
    if not phrase or not small_n:
        return False

    
    ref_phrase_words = phrase.strip().split()
    ref_len = len(ref_phrase_words)
    
    if ref_len < 3:
        return False
    first_word = ref_phrase_words[0]
    last_word = ref_phrase_words[-1]

    
    
    found_phrases = re.findall(rf"{re.escape(first_word)}.*?{re.escape(last_word)}", model_response)

    
    if len(found_phrases) != small_n:
        return False

    
    for found in found_phrases:
        current_words = found.split()

        
        if len(current_words) != ref_len:
            return False

        differences = 0
        diff_index = -1

        
        for i in range(ref_len):
            if current_words[i] != ref_phrase_words[i]:
                differences += 1
                diff_index = i

        
        
        if not (differences == 1 and 0 < diff_index < ref_len - 1):
            return False

    
    return True

def CopyChecker(model_response: str, parameters: dict) -> bool:
    """Run the `CopyChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    prompt_to_repeat = parameters.get("prompt_to_repeat")
    if not prompt_to_repeat:
        return False

    
    return model_response.strip().lower() == prompt_to_repeat.strip().lower()

def CopySpanIdxChecker(model_response: str, parameters: dict) -> bool:
    """Run the `CopySpanIdxChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    prompt_to_repeat = parameters.get("prompt_to_repeat")
    n_start_val = parameters.get("n_start")
    n_end_val = parameters.get("n_end")

    
    if prompt_to_repeat is None or n_start_val is None or n_end_val is None:
        return False

    try:
        
        n_start = int(n_start_val)
        n_end = int(n_end_val)
    except (ValueError, TypeError):
        
        
        return False

    
    
    if n_start < 0 or n_end > len(prompt_to_repeat) or n_start > n_end:
        
        return False
        
    expected_span = prompt_to_repeat[n_start:n_end]

    
    return model_response.strip().lower() == expected_span.strip().lower()

def SentenceHyphenChecker(model_response: str, parameters: dict) -> bool:
    """Run the `SentenceHyphenChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    
    def _split_into_sentences(text: str) -> list[str]:
        if not text:
            return []
        
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())
        
        return [s for s in sentences if s]

    
    hyphen_parts = model_response.split('-')

    
    
    
    if any(s.strip() != s for s in hyphen_parts):
        return False

    
    
    if len(hyphen_parts) > 1 and any(not s for s in hyphen_parts):
        return False

    
    reconstructed_text_with_spaces = " ".join(hyphen_parts)

    
    gold_standard_sentences = _split_into_sentences(reconstructed_text_with_spaces)

    
    
    return hyphen_parts == gold_standard_sentences

def AdjacentLetterChecker(model_response: str, parameters: dict) -> bool:
    """Run the `AdjacentLetterChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    words = model_response.split()
    if len(words) < 2:
        return True

    for i in range(len(words) - 1):
        
        if not words[i] or not words[i+1]:
            continue

        first_letter = words[i][0].lower()
        second_letter = words[i+1][0].lower()

        
        is_first_alpha = 'a' <= first_letter <= 'z'
        is_second_alpha = 'a' <= second_letter <= 'z'

        if is_first_alpha and is_second_alpha:
            
            if ord(second_letter) - ord(first_letter) == 1:
                return False
    return True

def SquareBracketChecker(model_response: str, parameters: dict) -> bool:
    """Run the `SquareBracketChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    words = model_response.split()
    
    if not words:
        return True
    for word in words:
        if not (word.startswith("[") and word.endswith("]")):
            return False
    return True

def KeywordFrequencyOnceChecker(model_response: str, parameters: dict) -> bool:
    """Run the `KeywordFrequencyOnceChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    keyword = parameters['keyword']
    actual_occurrences = len(re.findall(keyword, model_response, flags=re.IGNORECASE))
    return actual_occurrences == 1

def ExcludeWordHarderChecker(model_response: str, parameters: dict) -> bool:
    """Run the `ExcludeWordHarderChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    keyword = parameters["keyword"]
    
    if " " + keyword + " " in model_response:
        return False
    return True

def ParagraphBasicChecker(model_response: str, parameters: dict) -> bool:
    """Run the `ParagraphBasicChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    
    paragraphs = re.split(r"\s?\*\*\*\s?", model_response)
    num_paragraphs = len(paragraphs)

    
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():  
            if index == 0 or index == len(paragraphs) - 1:
                
                num_paragraphs -= 1
            else:
                
                return False

    
    return num_paragraphs == 2

def ParagraphBasicChecker2(model_response: str, parameters: dict) -> bool:
    """Run the `ParagraphBasicChecker2` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    
    paragraphs = re.split(r"\n\n", model_response)
    num_paragraphs = len(paragraphs)

    
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():  
            if index == 0 or index == len(paragraphs) - 1:
                
                num_paragraphs -= 1
            else:
                
                return False

    
    return num_paragraphs == 2

def FirstWordSentChecker(model_response: str, parameters: dict) -> bool:
    """Run the `FirstWordSentChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    expected_first_word = parameters.get("first_word")

    if not expected_first_word:
        
        return False

    
    if not isinstance(expected_first_word, str):
        expected_first_word = expected_first_word[0].strip()
    else:
        expected_first_word = expected_first_word.strip()

    sentences = instructions_util.split_into_sentences(model_response)

    
    if not sentences:
        return True

    
    for sentence in sentences:
        
        if not sentence.strip():
            return False
        
        words = sentence.split()
        
        if not words:
            return False

        first_word = words[0]
        
        if first_word.lower() != expected_first_word.lower():
            return False
            
    return True

def FirstWordAnswerChecker(model_response: str, parameters: dict) -> bool:
    """Run the `FirstWordAnswerChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    expected_first_word = parameters.get("first_word")

    
    if not expected_first_word:
        return False
    
    expected_first_word = expected_first_word.strip()

    
    if not model_response.strip():
        return False

    
    first_word = model_response.split()[0].strip(string.punctuation)

    
    if first_word.lower() != expected_first_word.lower():
        return False
        
    return True

def LastWordSentChecker(model_response: str, parameters: dict) -> bool:
    """Run the `LastWordSentChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    last_word = parameters.get("last_word")
    if not last_word:
        return False

    
    if not isinstance(last_word, str):
        last_word = str(last_word[0]).strip()
    else:
        last_word = last_word.strip()

    sentences = instructions_util.split_into_sentences(model_response)

    
    if not sentences:
        return False

    
    for sentence in sentences:
        stripped_sentence = sentence.strip()
        if not stripped_sentence:
            return False  

        words = stripped_sentence.split()
        if not words:
            return False 

        
        last_word_in_sentence = re.sub(r"[^\w\s]", "", words[-1])

        if last_word_in_sentence.lower() != last_word.lower():
            return False
            
    return True

def LastWordAnswerChecker(model_response: str, parameters: dict) -> bool:
    """Run the `LastWordAnswerChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    expected_last_word = parameters.get("last_word")
    if not expected_last_word:
        return False
    
    expected_last_word = str(expected_last_word).strip()

    words = model_response.split()
    if not words:
        return False

    
    actual_last_word = re.sub(r"[^\w\s]", "", words[-1])

    if actual_last_word.lower() != expected_last_word.lower():
        return False
        
    return True

def BiGramWrappingChecker(model_response: str, parameters: dict) -> bool:
    """Run the `BiGramWrappingChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    words = model_response.split()

    
    if not words:
        return True

    
    if len(words) % 2 != 0:
        return False

    
    for i in range(0, len(words), 2):
        part1 = words[i]
        part2 = words[i+1]

        
        is_part1_valid = (
            part1.startswith("<<") and
            len(part1) > 2 and  
            '>' not in part1 and
            '<' not in part1[2:]
        )

        
        is_part2_valid = (
            part2.endswith(">>") and
            len(part2) > 2 and  
            '<' not in part2 and
            '>' not in part2[:-2]
        )

        if not (is_part1_valid and is_part2_valid):
            return False

    return True

def CopyingSimpleChecker(model_response: str, parameters: dict) -> bool:
    """Run the `CopyingSimpleChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    prompt_to_repeat = parameters.get("prompt_to_repeat")

    
    if not prompt_to_repeat:
        return False

    
    return model_response.strip().lower() == str(prompt_to_repeat).strip().lower()

def CopyingMultipleChecker(model_response: str, parameters: dict) -> bool:
    """Run the `CopyingMultipleChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    prompt_to_repeat = parameters.get("prompt_to_repeat")
    N = parameters.get("N")

    
    if not prompt_to_repeat or not isinstance(N, int) or N <= 0:
        return False

    
    prompts = model_response.split("******")

    
    if len(prompts) != N:
        return False

    
    clean_prompt_to_repeat = str(prompt_to_repeat).strip().lower()

    
    for prompt in prompts:
        if prompt.strip().lower() != clean_prompt_to_repeat:
            return False
            
    return True

def PunctuationDotChecker(model_response: str, parameters: dict) -> bool:
    """Run the `PunctuationDotChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    return not re.search(r"\.", model_response)

def PunctuationExclamationChecker(model_response: str, parameters: dict) -> bool:
    """Run the `PunctuationExclamationChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    return not re.search(r"\!", model_response)

def LowercaseCountingChecker(model_response: str, parameters: dict) -> bool:
    """Run the `LowercaseCountingChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    N = parameters['N']
    lowercase_words = re.findall(r"\b[a-z]+\b", model_response)
    return len(lowercase_words) <= N

def LetterCountingChecker(model_response: str, parameters: dict) -> bool:
    """Run the `LetterCountingChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    N = parameters['N']
    relation = parameters['relation']
    letters = re.findall(r"[a-zA-Z]", model_response)
    
    if relation == "at least":
        return len(letters) >= N
    elif relation == "less than":
        return len(letters) < N
    
    return False

def CountingCompositionChecker(model_response: str, parameters: dict) -> bool:
    """Run the `CountingCompositionChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    n_sent = parameters['n_sent']
    n_words = parameters['n_words']
    
    
    paragraphs = re.split(r"\s?\*\s\*\s\*\s?", model_response)
    
    
    effective_paragraphs = []
    for p in paragraphs:
        if p.strip():
            effective_paragraphs.append(p)

    if len(effective_paragraphs) != 3:
        return False

    for paragraph in effective_paragraphs:
        
        sentences = instructions_util.split_into_sentences(paragraph)
        if len(sentences) != n_sent:
            return False

        for sentence in sentences:
            
            words = instructions_util.nltk.word_tokenize(sentence)
            if len(words) != n_words:
                return False

    return True

def CountUniqueChecker(model_response: str, parameters: dict) -> bool:
    """Run the `CountUniqueChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    
    words = instructions_util.nltk.word_tokenize(model_response)
    unique_words = set(words)
    return len(words) == len(unique_words)

def CountIncrementWordChecker(model_response: str, parameters: dict) -> bool:
    """Run the `CountIncrementWordChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    keyword1 = parameters.get("keyword1")
    keyword2 = parameters.get("keyword2")

    if not keyword1 or not keyword2:
        return False

    
    actual_occurrences1 = len(re.findall(keyword1, model_response, flags=re.IGNORECASE))
    actual_occurrences2 = len(re.findall(keyword2, model_response, flags=re.IGNORECASE))

    return actual_occurrences1 == 1 and actual_occurrences2 == 2


def PalindromeBasicChecker(model_response: str, parameters: dict) -> bool:
    """Run the `PalindromeBasicChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    
    
    palindromes = [word for word in model_response.split() if len(word) > 1 and word == word[::-1]]
    return len(palindromes) > 0

def KeywordSpecificPositionChecker(model_response: str, parameters: dict) -> bool:
    """Run the `KeywordSpecificPositionChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    keyword = parameters["keyword"]
    n = int(parameters["n"])
    m = int(parameters["m"])

    sentences = instructions_util.split_into_sentences(model_response)
    if len(sentences) < n:
        return False
    
    
    tokens = instructions_util.nltk.word_tokenize(sentences[n - 1])
    
    
    
    punctuation = string.punctuation + "，。！？【】（）《》“”‘’"
    words = [word for word in tokens if word not in punctuation]
    
    if len(words) < m:
        return False
    
    if words[m - 1] == keyword:
        return True
    else:
        return False

def StartEndChecker(model_response: str, parameters: dict) -> bool:
    """Run the `StartEndChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    words = instructions_util.nltk.word_tokenize(model_response)
    if len(words) < 2:
        return False
    
    return words[0].lower() == words[-1].lower()

def KeySentenceChecker(model_response: str, parameters: dict) -> bool:
    """Run the `KeySentenceChecker` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    key_sentences = parameters.get("key_sentences")
    num_sentences = parameters.get("num_sentences")
    relation = parameters.get("relation", "exact")

    if not key_sentences or num_sentences is None:
        return False

    
    response_sentences_cleaned = {s.strip() for s in instructions_util.split_into_sentences(model_response)}
    
    count = 0
    
    for sentence in key_sentences:
        
        if sentence.strip() in response_sentences_cleaned:
            count += 1

    
    if relation == 'at_least':
        return count >= num_sentences
    elif relation == 'at_most':
        return count <= num_sentences
    else:  
        return count == num_sentences

def KeywordFrequencyCheckerDifferent(model_response: str, parameters: dict) -> bool:
    """Run the `KeywordFrequencyCheckerDifferent` verification."""
    model_response = _filter_think_tags(model_response) # Remove the thinking block.
    
    keyword = parameters["keyword"]
    frequency = parameters["frequency"]
    relation = parameters["relation"]

    
    actual_occurrences = len(re.findall(keyword, model_response, flags=re.IGNORECASE))

    if relation == 'less than':
        return actual_occurrences < frequency
    elif relation == 'at least':
        return actual_occurrences >= frequency
    return False



# --- iFeval G ---





# VERIFICATION_FUNCTIONS = {
#     'math_verify': math_verify,
#     'word_count_range': word_count_range,
#     "PlaceholderChecker": PlaceholderChecker,
# }


VERIFICATION_FUNCTIONS ={
    '_verify_thinking_format': _verify_thinking_format,
    'math_verify': math_verify,
    'word_count_range': word_count_range,
    "ResponseLanguageChecker": ResponseLanguageChecker,
    "NumberOfSentences": NumberOfSentences,
    "PlaceholderChecker": PlaceholderChecker,
    "BulletListChecker": BulletListChecker,
    "ConstrainedResponseChecker": ConstrainedResponseChecker,
    "ConstrainedStartChecker": ConstrainedStartChecker,
    "HighlightSectionChecker": HighlightSectionChecker,
    "SectionChecker": SectionChecker,
    "ParagraphChecker": ParagraphChecker,
    "PostscriptChecker": PostscriptChecker,
    "RephraseChecker": RephraseChecker,
    "KeywordChecker": KeywordChecker,
    "KeywordFrequencyChecker": KeywordFrequencyChecker,
    "NumberOfWords": NumberOfWords,
    "JsonFormat": JsonFormat,
    "ParagraphFirstWordCheck": ParagraphFirstWordCheck,
    "ForbiddenWords": ForbiddenWords,
    "RephraseParagraph": RephraseParagraph,
    "TwoResponsesChecker": TwoResponsesChecker,
    "RepeatPromptThenAnswer": RepeatPromptThenAnswer,
    "EndChecker": EndChecker,
    "TitleChecker": TitleChecker,
    "LetterFrequencyChecker": LetterFrequencyChecker,
    "CapitalLettersEnglishChecker": CapitalLettersEnglishChecker,
    "LowercaseLettersEnglishChecker": LowercaseLettersEnglishChecker,
    "CommaChecker": CommaChecker,
    "CapitalWordFrequencyChecker": CapitalWordFrequencyChecker,
    "QuotationChecker": QuotationChecker,
    "RepeatPhraseChecker": RepeatPhraseChecker,
    "CopyChecker": CopyChecker,
    "CopySpanIdxChecker": CopySpanIdxChecker,
    "SentenceHyphenChecker": SentenceHyphenChecker,
    "AdjacentLetterChecker": AdjacentLetterChecker,
    "SquareBracketChecker": SquareBracketChecker,
    "KeywordFrequencyOnceChecker": KeywordFrequencyOnceChecker,
    "ExcludeWordHarderChecker": ExcludeWordHarderChecker,
    "ParagraphBasicChecker": ParagraphBasicChecker,
    "ParagraphBasicChecker2": ParagraphBasicChecker2,
    "FirstWordSentChecker": FirstWordSentChecker,
    "FirstWordAnswerChecker": FirstWordAnswerChecker,
    "LastWordSentChecker": LastWordSentChecker,
    "LastWordAnswerChecker": LastWordAnswerChecker,
    "BiGramWrappingChecker": BiGramWrappingChecker,
    "CopyingSimpleChecker": CopyingSimpleChecker,
    "CopyingMultipleChecker": CopyingMultipleChecker,
    "PunctuationDotChecker": PunctuationDotChecker,
    "PunctuationExclamationChecker": PunctuationExclamationChecker,
    "LowercaseCountingChecker": LowercaseCountingChecker,
    "LetterCountingChecker": LetterCountingChecker,
    "CountingCompositionChecker": CountingCompositionChecker,
    "CountUniqueChecker": CountUniqueChecker,
    "CountIncrementWordChecker": CountIncrementWordChecker,
    "PalindromeBasicChecker": PalindromeBasicChecker,
    "KeywordSpecificPositionChecker": KeywordSpecificPositionChecker,
    "StartEndChecker": StartEndChecker,
    "KeySentenceChecker": KeySentenceChecker,
    "KeywordFrequencyCheckerDifferent": KeywordFrequencyCheckerDifferent,
}


def get_verification_function(function_name: str):
    """Run the `get_verification_function` verification."""
    return VERIFICATION_FUNCTIONS.get(function_name)

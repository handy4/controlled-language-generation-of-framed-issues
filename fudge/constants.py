PAD_TOKEN = '[PAD]'
EOT_TOKEN = '<|endoftext|>'
SEP = 50256 # just use the weird eot token

TOPIC_MODEL_STRING = 'gpt2-medium'
FORMALITY_MODEL_STRING = 'Helsinki-NLP/opus-mt-es-en'

DIR_END_SPLIT_POSITIONS = 32

TOPIC_VAL_SIZE = 100000
FTOPIC_VAL_SIZE = 500
FORMALITY_VAL_SIZE = 2000
VOCAB_SIZE = 50000

FORMALITY_MAX_LEN = 200
ATTRIBUTE_MAX_LENGTH = 20

GLOVE_PRINT_PROGRESS_FREQ = 1000000
GLOVE_DIM = 300
HIDDEN_DIM = 300
RNN_DIM = 150

MIN_SENTENCE_LENGTH = 3

POETRY_LINE_SYLLABLES = 10
MAX_SYLLABLES_PER_WORD = 10 # no way anything is more
MAX_COUNT_SYLLABLE_DIST = 10
MAX_COUNT_SYLLABLE_INPUT_LENGTH = 25 # for just a couplet, shouldn't need more
COUNT_SYLLABLE_DIM = 100
UNKNOWN_RHYME_GROUP = 'UNKNOWN_RHYME_GROUP'
PHRASE_ENDS = '.?!'

POETRY_BANNED_TOKENS = [198, 50256, 628, 220] # newlines and eos and such


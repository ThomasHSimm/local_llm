"""Job Application Assistant — core pipeline modules."""
from .parser import parse_job_spec, parse_cv
from .analyser import analyse_gaps
from .elicitor import generate_questions
from .writer import draft_statement, count_words
from .session import save_session, load_session

from typing import List, Tuple, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class CreditCardRecognizer(PatternRecognizer):
    """
    Recognize common credit card numbers using regex + checksum.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    :param replacement_pairs: List of tuples with potential replacement values
    for different strings to be used during pattern matching.
    This can allow a greater variety in input, for example by removing dashes or spaces.
    """

    PATTERNS = [
        Pattern(
            "All Credit Cards (weak)",
            r"([출금]|[입금]|[잔액])", 
            0.3,
        ),
    ]

    CONTEXT = [
        "credit",
        "card",
        "visa",
        "mastercard",
        "cc ",
        "amex",
        "discover",
        "jcb",
        "diners",
        "maestro",
        "instapayment",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "CREDIT_CARD",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )


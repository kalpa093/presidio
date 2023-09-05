from typing import Optional, List

from presidio_analyzer import Pattern, PatternRecognizer


class IpRecognizer(PatternRecognizer):
    """
    Recognize Resident Registration Number using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    PATTERNS = [
        Pattern(
            "Resident Registration Number",
            r"\d{2}([0][1-9]|[1][0-2])([0][1-9]|[1-2]\d|[3][0-1])-?([1-4]{1})([0-9]{6})",
            0.1,
        ),
    ]

    CONTEXT = ["RNN", "Resident Registration Number", "주민등록번호"]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "IP_ADDRESS",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )


from typing import Optional, List

from presidio_analyzer import Pattern, PatternRecognizer


class KRMedInsRecognizer(PatternRecognizer):
    """
    Recognizes US driver license using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    PATTERNS = [
        Pattern(
            "Medical Insurance Number",
            r"[1257][-][0-9]{10}",
            0.1,
        ),
    ]
    CONTEXT = [
        "건강보험번호",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "건강보험번호",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )

import re

class SpeechBiomarkerExtractor:
    def __init__(self):
        pass

    def extract(self, transcript_text, segments, audio_duration):
        if not segments or audio_duration <= 0:
            return self._empty_stats()
            
        # Filter out trailing punctuation from text segments for word logic
        words = [re.sub(r'[^\w\s]', '', s['text']).strip().lower() for s in segments]
        words = [w for w in words if w]
        
        total_words = len(words)
        unique_words = len(set(words))
        
        # Pause logic (next_word.start - previous_word.end >= 0.4)
        pauses = []
        for i in range(1, len(segments)):
            prev_end = segments[i-1]['end']
            curr_start = segments[i]['start']
            gap = curr_start - prev_end
            if gap >= 0.4:
                pauses.append(gap)
                
        total_pause_duration = sum(pauses)
        pause_count = len(pauses)
        avg_pause = total_pause_duration / pause_count if pause_count > 0 else 0
        longest_pause = max(pauses) if pauses else 0
        
        # Audio duration from timestamps vs actual audio length
        speaking_duration = audio_duration - total_pause_duration
        if speaking_duration <= 0:
            speaking_duration = audio_duration
            
        wpm = (total_words / audio_duration) * 60 if audio_duration > 0 else 0
        speech_rate = (total_words / speaking_duration) * 60 if speaking_duration > 0 else 0
        pause_ratio = total_pause_duration / audio_duration if audio_duration > 0 else 0
        
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Approximate heuristics for prototypes
        hesitation_words = ["um", "uh", "ah", "er"]
        hesitation_count = sum(1 for w in words if w in hesitation_words)
        hesitation_ratio = hesitation_count / total_words if total_words > 0 else 0
        
        pronoun_words = ["he", "she", "it", "they", "them", "his", "her", "their", "him"]
        pronoun_count = sum(1 for w in words if w in pronoun_words)
        pronoun_ratio = pronoun_count / total_words if total_words > 0 else 0
        
        repeated_words = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1] and words[i] not in hesitation_words:
                repeated_words += 1
        repeat_ratio = repeated_words / total_words if total_words > 0 else 0
        
        # Sentences approx based on text transcript
        sentences = [s for s in re.split(r'[.!?]+', transcript_text) if s.strip()]
        sentence_count = max(1, len(sentences))
        avg_sentence_length = total_words / sentence_count
        
        fluency_score = max(0, min(100, 100 - (pause_ratio * 100) - (hesitation_ratio * 100) - (repeat_ratio * 100)))
        
        return {
            "words_per_minute": round(wpm, 1),
            "speech_rate": round(speech_rate, 1),
            "pause_ratio": round(pause_ratio, 2),
            "avg_pause": round(avg_pause, 2),
            "longest_pause": round(longest_pause, 2),
            "lexical_diversity": round(lexical_diversity, 2),
            "pronoun_ratio": round(pronoun_ratio, 2),
            "repeat_ratio": round(repeat_ratio, 2),
            "hesitation_ratio": round(hesitation_ratio, 2),
            "avg_sentence_length": round(avg_sentence_length, 1),
            "fluency_score": round(fluency_score, 0),
        }

    def _empty_stats(self):
        return {
            "words_per_minute": 0, "speech_rate": 0, "pause_ratio": 0,
            "avg_pause": 0, "longest_pause": 0, "lexical_diversity": 0,
            "pronoun_ratio": 0, "repeat_ratio": 0, "hesitation_ratio": 0,
            "avg_sentence_length": 0, "fluency_score": 0
        }

class ClinicalEvidenceScorer:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def score(self, biomarkers):
        evidence = []
        
        if biomarkers["pause_ratio"] >= self.thresholds["pause_ratio_high"]:
            evidence.append({"rank": "★★★★★", "feature": "Long pause ratio", "finding": f"{biomarkers['pause_ratio'] * 100:.1f}% of recording in pauses", "clinical_relevance": "May indicate word-retrieval latency or hesitation.", "severity_weight": 5})
        if biomarkers["speech_rate"] <= self.thresholds["speech_rate_low"]:
            evidence.append({"rank": "★★★★★", "feature": "Slow speech rate", "finding": f"{biomarkers['speech_rate']:.1f} WPM", "clinical_relevance": "May indicate reduced motor-linguistic processing speed.", "severity_weight": 5})
        if biomarkers["repeat_ratio"] >= self.thresholds["repeat_ratio_high"]:
            evidence.append({"rank": "★★★★☆", "feature": "Word repetition", "finding": f"Repeated words ratio at {biomarkers['repeat_ratio'] * 100:.1f}%", "clinical_relevance": "May indicate verbal perseveration.", "severity_weight": 4})
        if biomarkers["lexical_diversity"] <= self.thresholds["lexical_diversity_low"]:
            evidence.append({"rank": "★★★☆☆", "feature": "Reduced lexical diversity", "finding": f"Type-token ratio of {biomarkers['lexical_diversity']:.2f}", "clinical_relevance": "May reflect reduced active vocabulary in this sample.", "severity_weight": 3})
        if biomarkers["pronoun_ratio"] >= self.thresholds["pronoun_ratio_high"]:
            evidence.append({"rank": "★★★☆☆", "feature": "Elevated vague pronouns", "finding": f"Pronoun ratio of {biomarkers['pronoun_ratio'] * 100:.1f}%", "clinical_relevance": "May reflect reduced semantic specificity.", "severity_weight": 3})
        if biomarkers["avg_sentence_length"] <= self.thresholds["avg_sentence_length_low"]:
            evidence.append({"rank": "★★★☆☆", "feature": "Shorter sentences", "finding": f"Average sentence length of {biomarkers['avg_sentence_length']:.1f} words", "clinical_relevance": "Describes reduced utterance complexity in this sample.", "severity_weight": 3})
            
        return sorted(evidence, key=lambda item: item["severity_weight"], reverse=True)


class ClinicalExplanationGenerator:
    def generate(self, biomarkers, prediction, evidence):
        probability = prediction["probability"]
        severity = prediction["severity"]
        if not evidence:
            evidence_text = "No individual handcrafted speech biomarker exceeded the configured reference thresholds."
        else:
            evidence_text = "The strongest observed signals were " + ", ".join(item["feature"].lower() for item in evidence[:3]) + "."
        decision_text = "supports a positive dementia screening result" if prediction["positive"] else "does not support a positive dementia screening result"
        return (f"The model estimated a {probability * 100:.1f}% probability of dementia, which {decision_text} "
                f"at the configured threshold of {prediction['threshold']:.2f}. {evidence_text} "
                f"The estimated MMSE is {prediction['mmse']:.1f}/30 with an estimated severity of {severity}. "
                "This is an assistive research result and requires qualified clinical assessment.")


class ClinicalReportGenerator:
    def generate(self, prediction, biomarkers, evidence, explanation):
        lines = [
            "DEMENTIA SPEECH ASSESSMENT REPORT",
            "=================================",
            f"Decision       : {prediction['decision']}",
            f"Probability    : {prediction['probability'] * 100:.1f}%",
            f"Threshold      : {prediction['threshold']:.2f}",
            f"Estimated MMSE : {prediction['mmse']:.1f} / 30",
            f"Severity       : {prediction['severity']}",
            "",
            "SPEECH BIOMARKERS",
            f"Speech rate    : {biomarkers.get('speech_rate', 0):.1f} WPM",
            f"Pause ratio    : {biomarkers.get('pause_ratio', 0) * 100:.1f}%",
            f"Fluency score  : {biomarkers.get('fluency_score', 0):.1f} / 100",
            f"Lexical TTR    : {biomarkers.get('lexical_diversity', 0):.2f}",
            "",
            "CLINICAL EXPLANATION",
            explanation,
            "",
            "RANKED EVIDENCE",
        ]
        lines.extend(f"{item['rank']} {item['feature']}: {item['finding']} | {item['clinical_relevance']}" for item in evidence)
        lines.extend(["", "Disclaimer: This research output is not a standalone medical diagnosis."])
        return "\n".join(lines)

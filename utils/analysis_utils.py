import pandas as pd
from collections import Counter


def filter_dataset_by_emotion(emotion_df, emotion_of_interest="Doubt/Confusion"):
    emotion_df["filter_ix"] = emotion_df.apply(
        lambda x: emotion_of_interest in x["Categorical_Labels"], axis=1
    )
    emotion_df = emotion_df[emotion_df["filter_ix"] == 1]
    emotion_df.drop(["filter_ix"], axis=1)
    return emotion_df


def count_cooccurrences(annotation_df, emotion_of_interest="Doubt/Confusion"):
    list_list_emotions = annotation_df["Categorical_Labels"].tolist()
    emotion_list = [
        emotion for emotion_list in list_list_emotions for emotion in emotion_list
    ]
    emotion_counter = Counter(emotion_list)
    del emotion_counter[emotion_of_interest]
    return emotion_counter


def vad_summary_stats(
    annotation_df: pd.DataFrame, emotion_of_interest: str = "Doubt/Confusion"
):
    print(f"Summary stats for valence of emotion: {emotion_of_interest}")
    print(annotation_df["valence"].describe())
    print(f"Summary stats for arousal of emotion: {emotion_of_interest}")
    print(annotation_df["arousal"].describe())
    print(f"Summary stats for dominance of emotion: {emotion_of_interest}")
    print(annotation_df["dominance"].describe())

from automlstreams.evaluators import EvaluatePretrained
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tpot import TPOTClassifier

from datasets.streams import get_stream

BATCH_SIZE = 10000
MAX_SAMPLES = 10000


def run(model, topic:str):

    print(f'Running demo for file=/datasets/{topic}.csv')
    #stream = FileStream(f'/datasets/{topic}.csv')
    stream = get_stream(topic,from_cache=True)

    # Get a batch of BATCH_SIZE samples
    X, y = stream.next_sample(BATCH_SIZE)
    print('Sampled batch shape: ', X.shape)

    model.fit(X, y)

    model_name = model.__class__.__name__
    evaluator = EvaluatePretrained(show_plot=False,
                                   n_wait=200,
                                   batch_size=1,
                                   max_samples=MAX_SAMPLES,
                                   output_file=f'./results/batch.{model_name}.{topic}.csv')

    evaluator.evaluate(stream=stream, model=model)


if __name__ == "__main__":
    topics = [
        'agrawal_gen',
        'stagger_gen',
        'hyperplane_gen',
        'led_gen',
        'rbf_gen',
        'sea_gen',
        'covtype',
        'elec',
        'pokerhand'
    ]

    models = [
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        LinearSVC(),
        TPOTClassifier(generations=10,population_size=5),
        #AutoSklearnClassifier(per_run_time_limit=120)
    ]

    print([m.__class__.__name__ for m in models])
    for topic in topics:
        for model in models:
            print('\n', model.__class__.__name__, topic, ':\n')
            run(model, topic)

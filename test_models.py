import unittest as ut
from data_parsing import create_dataframe, parse_app_data, create_window_dataframe
from generic_markov import generic_markov


class TestMethods (ut.TestCase):

    def test_matrixConstruction(self):

        df_travel = create_dataframe()
        df_travel = df_travel.sample(frac=1)  # randomize the dataframe order, for consistency
        df_train = df_travel.sample(frac=0.7)
        df_test = df_travel.drop(df_train.index)

        mk_true = generic_markov(df=df_travel)
        mk_test = generic_markov(df=df_train)

        for r_id in range(len(df_test)):
            mk_test.fit(df_test.iloc[[r_id]])

        self.assertTrue(mk_true.transitionMatrix.all() == mk_test.transitionMatrix.all())

    def test_isWindowClustersEven(self):
        df_app = parse_app_data('/home/celadodc-rswl.com/corentin.tatger/Documents/data_1649679870505.jsonl')
        df_w = create_window_dataframe(df_app, verbose=False)
        nb_clusters = len(df_w['Coord_cluster'].unique())
        self.assertTrue(nb_clusters % 2 == 0 or nb_clusters == 1)

    def test_dataframeSizeUpdate(self):
        df_app = parse_app_data('/home/celadodc-rswl.com/corentin.tatger/Documents/data_1649679870505.jsonl')
        df_w = create_window_dataframe(df_app, verbose=False)
        mk = generic_markov(df=df_w)
        starting_length = len(mk.data_frame)

        df_app2 = parse_app_data('/home/celadodc-rswl.com/corentin.tatger/Documents/jsonl/data_1649213014018.jsonl')
        df_w2 = create_window_dataframe(df_app2, verbose=False)
        mk.fit(df_w2)
        # for r_id in range(len(df_w2)):
        #    mk.fit(df_w2.iloc[[r_id]])
        self.assertEqual(len(mk.data_frame), starting_length+len(df_w2))


if __name__ == "__main__":
    ut.main()

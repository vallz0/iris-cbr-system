import pycbr
import pandas as pd
from sklearn import datasets
import tempfile

class IrisCBR:
    def __init__(self):
        self.df = self._load_data()
        self.case_base = self._create_case_base()
        self.recovery = self._create_recovery()
        self.aggregation = self._create_aggregation()
        self.cbr = self._create_cbr()
        self.app = self.cbr.app

    def _load_data(self) -> pd.DataFrame:
        iris = datasets.load_iris()
        df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
        df['species'] = iris['target']
        df['species'] = df['species'].apply(lambda x: iris['target_names'][x])
        return df

    def _create_case_base(self) -> pycbr.casebase.SimpleCSVCaseBase:
        temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.df.to_csv(temp_file.name, index=False)
        return pycbr.casebase.SimpleCSVCaseBase(temp_file.name)

    def _create_recovery(self) -> pycbr.recovery.Recovery:
        return pycbr.recovery.Recovery(
            [(feature, pycbr.models.QuantileLinearAttribute()) for feature in self.df.columns[:-1]]
        )

    def _create_aggregation(self) -> pycbr.aggregate.MajorityAggregate:
        return pycbr.aggregate.MajorityAggregate('species')

    def _create_cbr(self) -> pycbr.CBR:
        return pycbr.CBR(self.case_base, self.recovery, self.aggregation, server_name='Iris-demo')

    def run(self):
        self.app.run()


def main() -> None:
    iris_cbr = IrisCBR()
    iris_cbr.run()

if __name__ == '__main__':
    main()

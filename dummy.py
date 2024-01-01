from lightning_data_terminal import DataModuleTerminal
from lightning_data_terminal import LightningCowDataModule
from lightning_data_terminal import MultiCamCowsDataModule

if __name__ == '__main__':
    # terminal = DataModuleTerminal(name="OpenSetCows2023",
    #                               current_fold=0,
    #                               folds_file="datasets/OpenSetCows2023/splits/10-90.json")
    terminal = MultiCamCowsDataModule(name="MultiCamCows",
                                      imsize=224,
                                      batch_size=16,
                                      num_workers=4)

    print(terminal)

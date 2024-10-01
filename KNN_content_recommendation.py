import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors

mainDf = pd.read_csv("datasets/cleanedAnime.csv")
fitDf = pd.read_csv("datasets/fitSet.csv")

actualNames = list(mainDf.Name.values)


def contentRec(name, type=None):
    if type:
        found_id = mainDf[mainDf["Name"] == name].index[0]
        for id in indices[found_id][1:]:
            dfToListDf = mainDf.loc[
                (mainDf["Unnamed: 0"] == id) & (mainDf["Type"] == type)
            ]
            for item in dfToListDf["Name"].values.tolist():
                print(item)
    else:
        found_id = mainDf[mainDf["Name"] == name].index[0]
        for id in indices[found_id][1:]:
            print(mainDf.iloc[id]["Name"])


while True:
    name = input("Enter the name of the anime: ")

    try:
        number = int(input("Enter number of recommendations: "))
    except ValueError:
        print("\033[1;3m" + "\033[91m" + "Input must be a number. \n" + "\033[0m")
        continue

    print("TV:1 | Movie:2 | Music:3 | OVA:4 | Web:5 | None:0")
    try:
        type_ = int(input("Enter number for type of anime: "))
    except ValueError:
        print("\033[1;3m" + "\033[91m" + "Input must be a number. \n" + "\033[0m")
        continue
    print()

    # KDtree is used because it is faster in low dimensional datasets
    nbrs = NearestNeighbors(n_neighbors=number, algorithm="kd_tree").fit(fitDf)
    distances, indices = nbrs.kneighbors(fitDf)

    try:
        match type_:
            case 0:
                print("\033[4m" + "Recommendations:" + "\033[0m")
                contentRec(name)
            case 1:
                print("\033[4m" + "Recommended TV Shows:" + "\033[0m")
                contentRec(name, "TV   ")
            case 2:
                print("\033[4m" + "Recommended Movies:" + "\033[0m")
                contentRec(name, "Movie")
            case 3:
                print("\033[4m" + "Recommended Music:" + "\033[0m")
                contentRec(name, "Music")
            case 4:
                print("\033[4m" + "Recommended OVA's:" + "\033[0m")
                contentRec(name, "OVA  ")
            case 5:
                print("\033[4m" + "Recommended Web Shows:" + "\033[0m")
                contentRec(name, "Web  ")
            case _:
                print("\033[1;3m" + "\033[91m" + "Invalid Input" + "\033[0m")
    except IndexError:
        print(
            "\033[1;3m"
            + "\033[91m"
            + f'"{name}" was not present in the available dataset'
            + "\033[0m"
            + "\033[1;3m"
            + "\033[4m"
            + f'\n \nAvailable names containing "{name}":'
            + "\033[0m"
        )
        for actualName in actualNames:
            if name in actualName:
                print(actualName)

        again = input("\nwould you like to go again?(y|n): ")
        if again == "n":
            break
        else:
            print()
            continue

    again = input("\nwould you like to go again(y|n): ")
    if again == "n":
        break
    print()

print("\033[92m" + "==========DONE==========" + "\033[0m")

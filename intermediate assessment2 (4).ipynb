{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "29176936-7b2a-4650-9135-80db2d67f85f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ===== 1. Import Libraries =====\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# ===== 2. Load Dataset =====\n",
    "train = pd.read_csv(\"train_LZdllcl (2).csv\")\n",
    "test = pd.read_csv(\"test_2umaH9m (2).csv\")\n",
    "sample = pd.read_csv(\"sample_submission_M0L0uXE (2).csv\")\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e273d146-763e-4d27-8343-b282268cfaba",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train Shape: (54808, 14)\n",
      "Test Shape: (23490, 13)\n",
      "   employee_id         department     region         education gender  \\\n",
      "0        65438  Sales & Marketing   region_7  Master's & above      f   \n",
      "1        65141         Operations  region_22        Bachelor's      m   \n",
      "2         7513  Sales & Marketing  region_19        Bachelor's      m   \n",
      "3         2542  Sales & Marketing  region_23        Bachelor's      m   \n",
      "4        48945         Technology  region_26        Bachelor's      m   \n",
      "\n",
      "  recruitment_channel  no_of_trainings  age  previous_year_rating  \\\n",
      "0            sourcing                1   35                   5.0   \n",
      "1               other                1   30                   5.0   \n",
      "2            sourcing                1   34                   3.0   \n",
      "3               other                2   39                   1.0   \n",
      "4               other                1   45                   3.0   \n",
      "\n",
      "   length_of_service  KPIs_met >80%  awards_won?  avg_training_score  \\\n",
      "0                  8              1            0                  49   \n",
      "1                  4              0            0                  60   \n",
      "2                  7              0            0                  50   \n",
      "3                 10              0            0                  50   \n",
      "4                  2              0            0                  73   \n",
      "\n",
      "   is_promoted  \n",
      "0            0  \n",
      "1            0  \n",
      "2            0  \n",
      "3            0  \n",
      "4            0  \n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 54808 entries, 0 to 54807\n",
      "Data columns (total 14 columns):\n",
      " #   Column                Non-Null Count  Dtype  \n",
      "---  ------                --------------  -----  \n",
      " 0   employee_id           54808 non-null  int64  \n",
      " 1   department            54808 non-null  object \n",
      " 2   region                54808 non-null  object \n",
      " 3   education             52399 non-null  object \n",
      " 4   gender                54808 non-null  object \n",
      " 5   recruitment_channel   54808 non-null  object \n",
      " 6   no_of_trainings       54808 non-null  int64  \n",
      " 7   age                   54808 non-null  int64  \n",
      " 8   previous_year_rating  50684 non-null  float64\n",
      " 9   length_of_service     54808 non-null  int64  \n",
      " 10  KPIs_met >80%         54808 non-null  int64  \n",
      " 11  awards_won?           54808 non-null  int64  \n",
      " 12  avg_training_score    54808 non-null  int64  \n",
      " 13  is_promoted           54808 non-null  int64  \n",
      "dtypes: float64(1), int64(8), object(5)\n",
      "memory usage: 5.9+ MB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(\"Train Shape:\", train.shape)\n",
    "print(\"Test Shape:\", test.shape)\n",
    "print(train.head())\n",
    "print(train.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "76d60c5f-faf3-44b9-b4a0-bc3f3af92423",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "employee_id                0\n",
      "department                 0\n",
      "region                     0\n",
      "education               2409\n",
      "gender                     0\n",
      "recruitment_channel        0\n",
      "no_of_trainings            0\n",
      "age                        0\n",
      "previous_year_rating    4124\n",
      "length_of_service          0\n",
      "KPIs_met >80%              0\n",
      "awards_won?                0\n",
      "avg_training_score         0\n",
      "is_promoted                0\n",
      "dtype: int64\n",
      "employee_id                0\n",
      "department                 0\n",
      "region                     0\n",
      "education               1034\n",
      "gender                     0\n",
      "recruitment_channel        0\n",
      "no_of_trainings            0\n",
      "age                        0\n",
      "previous_year_rating    1812\n",
      "length_of_service          0\n",
      "KPIs_met >80%              0\n",
      "awards_won?                0\n",
      "avg_training_score         0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(train.isnull().sum())\n",
    "print(test.isnull().sum())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "2e861f80-9e95-46a1-b96c-ae25db54bb7e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill missing numerical values with median\n",
    "train['previous_year_rating'] = train['previous_year_rating'].fillna(train['previous_year_rating'].median())\n",
    "test['previous_year_rating'] = test['previous_year_rating'].fillna(test['previous_year_rating'].median())\n",
    "\n",
    "train['education'] = train['education'].fillna(train['education'].mode()[0])\n",
    "test['education'] = test['education'].fillna(test['education'].mode()[0])\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "f7d79800-cba2-4b52-b789-8c2d28ef7358",
   "metadata": {},
   "outputs": [],
   "source": [
    "categorical_cols = train.select_dtypes(include=['object']).columns\n",
    "\n",
    "le = LabelEncoder()\n",
    "for col in categorical_cols:\n",
    "    train[col] = le.fit_transform(train[col])\n",
    "    test[col] = le.transform(test[col])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ce07a820-d510-43a8-b76d-7c523796cb7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = train.drop([\"is_promoted\", \"employee_id\"], axis=1)\n",
    "y = train[\"is_promoted\"]\n",
    "\n",
    "# train-validation split\n",
    "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "cc296cb4-ef4b-4eaf-84e9-fa04491fe4f4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "F1 Score: 0.42483660130718953\n"
     ]
    }
   ],
   "source": [
    "model = RandomForestClassifier(n_estimators=300, random_state=42)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Validation Check\n",
    "y_pred = model.predict(X_val)\n",
    "print(\"F1 Score:\", f1_score(y_val, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "7f488e55-e21e-4743-874f-e3ce69aad9fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(X, y)\n",
    "test_predictions = model.predict(test.drop(\"employee_id\", axis=1))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "e516236c-5019-439d-b4fe-77b4a85b6832",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ✅ Create submission file correctly\n",
    "submission = sample.copy()\n",
    "submission['is_promoted'] = test_predictions\n",
    "submission.to_csv(\"final_submission.csv\", index=False)\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "89cebe38-4a4a-4b8a-9e5e-f3ecc8b5ca37",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['employee_id', 'is_promoted'], dtype='object')\n",
      "   employee_id  is_promoted\n",
      "0         8724            0\n",
      "1        74430            0\n",
      "2        72255            0\n",
      "3        38562            0\n",
      "4        64486            0\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv(\"final_submission.csv\")\n",
    "print(df.columns)\n",
    "print(df.head())\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "c457da98-a9d7-41cf-ba4a-bd0c59cd8545",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<a href='final_submission.csv' target='_blank'>final_submission.csv</a><br>"
      ],
      "text/plain": [
       "C:\\Users\\EMIL MATHEW\\final_submission.csv"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import FileLink\n",
    "FileLink(\"final_submission.csv\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "be177dbc-23a4-4560-88f7-3ef0d85be11e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b90cb5ec-a749-44d5-82fd-dcb84123870d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

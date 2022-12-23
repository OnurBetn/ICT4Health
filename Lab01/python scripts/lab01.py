from sub.minimization import *
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    np.random.seed(46)

    # Read the data from the file, shuffle the rows and reset the index column
    data = pd.read_csv('parkinsons_updrs.data').sample(frac=1).reset_index(drop=True)
    # Removing the useless features (like subject id, sex, etc.)
    data.drop(['subject#', 'age', 'sex', 'test_time'], axis=1, inplace=True)
    # Number of records
    Nr = len(data)

    # Training set
    data_train = data.iloc[:int(Nr / 2)]
    # Validation set
    data_val = data.iloc[int(Nr / 2):int(Nr - Nr / 4)]
    # Testing set
    data_test = data.iloc[int(Nr - Nr / 4):Nr]

    # Normalizing the data
    data_train_norm = (data_train - data_train.mean()) / data_train.std()
    data_val_norm = (data_val - data_train.mean()) / data_train.std()
    data_test_norm = (data_test - data_train.mean()) / data_train.std()

    F0 = 'total_UPDRS'  # Regressand

    # Defining matrices X's and vectors y's 
    X_train_norm = data_train_norm.drop(F0, axis=1).values
    y_train_norm = data_train_norm.loc[:, F0].values
    y_train_norm.shape = (y_train_norm.size, 1)  # Adjusting shape from (N,) to (N,1)

    X_test_norm = data_test_norm.drop(F0, axis=1).values
    y_test_norm = data_test_norm.loc[:, F0].values
    y_test_norm.shape = (y_test_norm.size, 1)  # Adjusting shape from (N,) to (N,1)

    X_val_norm = data_val_norm.drop(F0, axis=1).values
    y_val_norm = data_val_norm.loc[:, F0].values
    y_val_norm.shape = (y_val_norm.size, 1)  # Adjusting shape from (N,) to (N,1)

    # Menu to choose the algorithm
    while True:
        print('\n-------------------------------------------')
        print('\t 1. LLS pseudoinverse')
        print('\t 2. Conjugate gradient')
        print('\t 3. Gradient algorihm')
        print('\t 4. Stochastic gradient')
        print('\t 5. Steepest descent')
        print('\t 6. Ridge regression')
        print('\t 0. Exit')
        command = int(input('Please choose the algorithm to perform -> '))
        if command == 1:
            title = 'LLS pseudoinverse'
            m = SolveLLS(y_train_norm, X_train_norm)
        elif command == 2:
            title = 'Conjugate gradient'
            m = SolveConjGrad(y_train_norm, X_train_norm)
        elif command == 3:
            title = 'Gradient algorihm'
            m = SolveGrad(y_train_norm, X_train_norm, y_val_norm, X_val_norm, gamma=1e-5, Nit=10000)
        elif command == 4:
            title = 'Stochastic gradient'
            m = SolveStochGrad(y_train_norm, X_train_norm, y_val_norm, X_val_norm, gamma=1e-3, Nit=100000)
        elif command == 5:
            title = 'Steepest descent'
            m = SolveSteepDesc(y_train_norm, X_train_norm, y_val_norm, X_val_norm, Nit=1000)
        elif command == 6:
            title = 'Ridge regression'
            m = SolveRidge(y_train_norm, X_train_norm, y_test_norm, X_test_norm)
        elif command == 0:
            break
        else:
            print('\nThis is not a valid choice. Please choose again\n')
            continue

        w_hat = m.run()

        # Un-normalizing X's and y's
        y_train = y_train_norm * data_train[F0].std() + data_train[F0].mean()
        y_test = y_test_norm * data_train[F0].std() + data_train[F0].mean()
        y_val = y_val_norm * data_train[F0].std() + data_train[F0].mean()
        yhat_train = np.dot(X_train_norm, w_hat) * data_train[F0].std() + data_train[F0].mean()
        yhat_test = np.dot(X_test_norm, w_hat) * data_train[F0].std() + data_train[F0].mean()
        yhat_val = np.dot(X_val_norm, w_hat) * data_train[F0].std() + data_train[F0].mean()

        # Plots of yhat's vs y's
        m.plot_y(yhat_train, y_train, 'yhat_train', 'y_train', title)
        m.plot_y(yhat_test, y_test, 'yhat_test', 'y_test', title)

        # Plots of the histograms of y-y_hat
        m.hist_y(yhat_train, y_train, 'yhat_train', 'y_train', title)
        m.hist_y(yhat_test, y_test, 'yhat_test', 'y_test', title)

        # Print and plot the weight vector
        m.print_result(title)
        m.plot_w(title + ': weight vector')

        # Plot of the error
        m.plot_err(title + ': mean square error', logy=1, logx=0)

        # Print the mean square errors
        print('Mean square error for the training set: ', round(mean_squared_error(y_train, yhat_train), 5))
        print('Mean square error for the testing set: ', round(mean_squared_error(y_test, yhat_test), 5))
        print('Mean square error for the validation set: ', round(mean_squared_error(y_val, yhat_val), 5))

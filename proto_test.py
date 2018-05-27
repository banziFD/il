



# Construct Examplar Set and save it as a dict
for i in range(20):
        protoset = dict()
        print('Constructing protoset')
        protoset = icarl.construct_proto(iter_group, mixing, loader, protoset)

        protoset_name = work_path + '/protoset_{}_{}'.format(iter_group, i)

        with open(protoset_name, 'wb') as f:
            pickle.dump(protoset, f)
            f.close()
        print('Complete protoset')
        print('Testing')
        testset = utils_data.MyDataset(work_path, 0, 2)
        testloader = DataLoader(testset, batch_size = batch_size, shuffle = True)
        icarl.feature_extract(testloader, test_path, iter_group)
        icarl.classify(protoset, test_path, iter_group)
        print('Complete test')
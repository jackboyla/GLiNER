import torch
from rel_rep import get_entity_pair_reps
from base import generate_entity_pairs_indices


def test_forward_output_matches_data_processing():
    ''' 
    NOTE:   the input for get_entity_pair_reps() is 
            entity representation of shape ([B, number_of_entities, D])

            the input for generate_entity_pairs_indices() is
            a single instance of shape [number_of_entities, 2]

            for this test, we will be using input of shape ([B, number_of_entities, 2]) 
            This let's us visually inspect that the two methods are generating pairs of 
            entities/spans in the same order

    '''
    
    # Mock input data
    # must be same num_entities per example for rel_rep (because it expects padded input)
    span_idx = torch.tensor(
        [
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], 
            [[21, 21], [22, 22], [23, 23], [24, 24], [25, 25]], 
            [[800, 800], [801, 801], [0, 0], [0, 0], [0, 0]],
            [[900, 900], [901, 901], [902, 902], [903, 903], [904, 904]],
        ]
    )      
    #  -->  ([B, number_of_entities, 2])  

    batch_size = span_idx.size(0)

    
    # Run the model's forward method for generating pairs
    rel_rep = get_entity_pair_reps(span_idx) #  -->  ([B, num_unique_pairs, 4])
    
    # Compute pairs of entities as done during precprocessing
    base_pairs_list = []
    batch_pairs_list = []
    for batch_i in range(batch_size):
        batch_pairs = generate_entity_pairs_indices(span_idx[batch_i])
        # batch_pairs  -->  ([num_unique_pairs, 2 ->start_index, 2 ->end_index])
        batch_pairs_list.append(batch_pairs)

        num_unique_pairs = batch_pairs.size(0)
        reshaped_batch_pairs = batch_pairs.reshape(num_unique_pairs, -1)
        # reshaped_batch_pairs  -->  ([num_unique_pairs, 4])

        base_pairs_list.append(reshaped_batch_pairs.tolist())

    base_pairs = torch.tensor(base_pairs_list)  # -->  ([B, num_unique_pairs, 4])

    
    # Assert the outputs are equal
    assert rel_rep.equal(base_pairs), "get_entity_pair_reps and generate_entity_pairs_indices do not give the same output"


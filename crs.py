import numpy as np
from scipy.special import erfinv

def generate_10_single(n):

    #
    # Half width as random number always +ve
    #

    ntrial = n
    step_size = 1./ntrial
    random_numbers=[]

    #
    # +ve case
    #

    for i in range(ntrial):

        rno = i*step_size + step_size*np.random.random()
        random_numbers.append(np.sqrt(2.)*erfinv(rno))

    #
    # -ve case
    #

    for i in range(ntrial):

        rno = i*step_size + step_size*np.random.random()
        random_numbers.append(-np.sqrt(2.)*erfinv(rno))

    return random_numbers

def generate_10(n):

    rand1 = generate_10_single(n)
    rand2 = generate_10_single(n)
    rand3 = generate_10_single(n)

    dist1 = np.mean(rand1)**2 + (np.std(rand1)-1.)**2
    dist2 = np.mean(rand2)**2 + (np.std(rand2)-1.)**2
    dist3 = np.mean(rand3)**2 + (np.std(rand3)-1.)**2

    if dist1 < dist2 and dist1 < dist3:
        random_numbers = np.copy(rand1)
    elif dist2 < dist1 and dist2 < dist3:
        random_numbers = np.copy(rand2)
    elif dist3 < dist1 and dist3 < dist2:
        random_numbers = np.copy(rand3)

    #
    # Now randomise the numbers (mix them up)
    #

    in_index = np.arange(2*n).astype(dtype=np.int32)
    in_index_rand = np.random.random(size=(2*n))
    sort_index = np.argsort(in_index_rand)
    index = in_index[sort_index]
    random_numbers = random_numbers[index]

    return random_numbers


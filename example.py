import numpy as np
from des import des_encrypt, des_decrypt


plain_text = np.array([
    int(bit)
    for bit in np.binary_repr(0x8787878787878787, width=64)
], dtype=np.uint8)

print('plain text:', plain_text)


key = np.array([
    int(bit)
    for bit in np.binary_repr(0x0E329232EA6D0D73, width=64)
], dtype=np.uint8)

cipher_text = des_encrypt(plain_text, key)
print('cipher text:', cipher_text)

decrypted_plain_text = des_decrypt(cipher_text, key)
print('decrypted plain text:', decrypted_plain_text)

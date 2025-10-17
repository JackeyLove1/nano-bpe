"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from nanobpe import BasicTokenizer, RegexTokenizer

# open some text and train a vocab of 512 tokens
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

# construct the Tokenizer object and kick off verbose training
tokenizer = BasicTokenizer()
t0 = time.time()
tokenizer.train(text, 512, verbose=True)
# writes two files in the models directory: name.model, and name.vocab
t1 = time.time()
prefix = os.path.join("models", "BasicTokenizer")
tokenizer.save(prefix)


print(f"Training took {t1 - t0:.2f} seconds")
'''
# device info
Model name:            Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
Thread(s) per core:  2
Core(s) per socket:  24
Frequency boost:     enabled
CPU max MHz:         2501.0000
CPU min MHz:         1000.0000

# train log
merge 1/256: (101, 32) -> 256 (b'e ') had 2981 occurrences
merge 2/256: (44, 32) -> 257 (b', ') had 2961 occurrences
merge 3/256: (100, 32) -> 258 (b'd ') had 2617 occurrences
merge 4/256: (46, 32) -> 259 (b'. ') had 2560 occurrences
merge 5/256: (114, 32) -> 260 (b'r ') had 2428 occurrences
merge 6/256: (50, 48) -> 261 (b'20') had 2365 occurrences
merge 7/256: (115, 32) -> 262 (b's ') had 2053 occurrences
merge 8/256: (105, 110) -> 263 (b'in') had 2006 occurrences
merge 9/256: (111, 110) -> 264 (b'on') had 1815 occurrences
merge 10/256: (114, 105) -> 265 (b'ri') had 1805 occurrences
merge 11/256: (116, 32) -> 266 (b't ') had 1802 occurrences
merge 12/256: (116, 104) -> 267 (b'th') had 1737 occurrences
merge 13/256: (101, 258) -> 268 (b'ed ') had 1736 occurrences
merge 14/256: (257, 261) -> 269 (b', 20') had 1705 occurrences
merge 15/256: (97, 110) -> 270 (b'an') had 1487 occurrences
merge 16/256: (97, 114) -> 271 (b'ar') had 1360 occurrences
merge 17/256: (101, 260) -> 272 (b'er ') had 1356 occurrences
merge 18/256: (121, 32) -> 273 (b'y ') had 1248 occurrences
merge 19/256: (97, 108) -> 274 (b'al') had 1164 occurrences
merge 20/256: (267, 256) -> 275 (b'the ') had 1142 occurrences
merge 21/256: (118, 268) -> 276 (b'ved ') had 1104 occurrences
merge 22/256: (119, 105) -> 277 (b'wi') had 1049 occurrences
merge 23/256: (101, 114) -> 278 (b'er') had 897 occurrences
merge 24/256: (264, 32) -> 279 (b'on ') had 880 occurrences
merge 25/256: (277, 102) -> 280 (b'wif') had 871 occurrences
merge 26/256: (82, 101) -> 281 (b'Re') had 870 occurrences
merge 27/256: (83, 280) -> 282 (b'Swif') had 867 occurrences
merge 28/256: (111, 260) -> 283 (b'or ') had 859 occurrences
merge 29/256: (99, 104) -> 284 (b'ch') had 816 occurrences
merge 30/256: (269, 49) -> 285 (b', 201') had 811 occurrences
merge 31/256: (111, 109) -> 286 (b'om') had 789 occurrences
merge 32/256: (98, 272) -> 287 (b'ber ') had 752 occurrences
merge 33/256: (32, 275) -> 288 (b' the ') had 748 occurrences
merge 34/256: (97, 121) -> 289 (b'ay') had 744 occurrences
merge 35/256: (101, 110) -> 290 (b'en') had 740 occurrences
merge 36/256: (111, 114) -> 291 (b'or') had 737 occurrences
merge 37/256: (274, 32) -> 292 (b'al ') had 705 occurrences
merge 38/256: (101, 109) -> 293 (b'em') had 703 occurrences
merge 39/256: (46, 10) -> 294 (b'.\n') had 685 occurrences
merge 40/256: (265, 101) -> 295 (b'rie') had 685 occurrences
merge 41/256: (263, 103) -> 296 (b'ing') had 684 occurrences
merge 42/256: (269, 50) -> 297 (b', 202') had 673 occurrences
merge 43/256: (116, 105) -> 298 (b'ti') had 666 occurrences
merge 44/256: (289, 108) -> 299 (b'ayl') had 654 occurrences
merge 45/256: (34, 259) -> 300 (b'". ') had 651 occurrences
merge 46/256: (108, 108) -> 301 (b'll') had 649 occurrences
merge 47/256: (84, 299) -> 302 (b'Tayl') had 647 occurrences
merge 48/256: (116, 295) -> 303 (b'trie') had 644 occurrences
merge 49/256: (294, 32) -> 304 (b'.\n ') had 643 occurrences
merge 50/256: (116, 111) -> 305 (b'to') had 642 occurrences
merge 51/256: (259, 281) -> 306 (b'. Re') had 640 occurrences
merge 52/256: (306, 303) -> 307 (b'. Retrie') had 639 occurrences
merge 53/256: (307, 276) -> 308 (b'. Retrieved ') had 639 occurrences
merge 54/256: (302, 283) -> 309 (b'Taylor ') had 611 occurrences
merge 55/256: (101, 115) -> 310 (b'es') had 606 occurrences
merge 56/256: (309, 282) -> 311 (b'Taylor Swif') had 598 occurrences
merge 57/256: (117, 115) -> 312 (b'us') had 561 occurrences
merge 58/256: (114, 286) -> 313 (b'rom') had 532 occurrences
merge 59/256: (293, 287) -> 314 (b'ember ') had 528 occurrences
merge 60/256: (41, 259) -> 315 (b'). ') had 524 occurrences
merge 61/256: (65, 114) -> 316 (b'Ar') had 509 occurrences
merge 62/256: (102, 313) -> 317 (b'from') had 503 occurrences
merge 63/256: (315, 34) -> 318 (b'). "') had 499 occurrences
merge 64/256: (270, 258) -> 319 (b'and ') had 498 occurrences
merge 65/256: (114, 101) -> 320 (b're') had 495 occurrences
merge 66/256: (111, 117) -> 321 (b'ou') had 487 occurrences
merge 67/256: (111, 265) -> 322 (b'ori') had 469 occurrences
merge 68/256: (111, 102) -> 323 (b'of') had 466 occurrences
merge 69/256: (103, 263) -> 324 (b'gin') had 465 occurrences
merge 70/256: (296, 32) -> 325 (b'ing ') had 464 occurrences
merge 71/256: (284, 105) -> 326 (b'chi') had 458 occurrences
merge 72/256: (93, 32) -> 327 (b'] ') had 458 occurrences
merge 73/256: (324, 292) -> 328 (b'ginal ') had 453 occurrences
merge 74/256: (317, 288) -> 329 (b'from the ') had 447 occurrences
merge 75/256: (322, 328) -> 330 (b'original ') had 446 occurrences
merge 76/256: (104, 256) -> 331 (b'he ') had 440 occurrences
merge 77/256: (316, 326) -> 332 (b'Archi') had 440 occurrences
merge 78/256: (332, 276) -> 333 (b'Archived ') had 440 occurrences
merge 79/256: (329, 330) -> 334 (b'from the original ') had 440 occurrences
merge 80/256: (333, 334) -> 335 (b'Archived from the original ') had 439 occurrences
merge 81/256: (335, 279) -> 336 (b'Archived from the original on ') had 438 occurrences
merge 82/256: (259, 336) -> 337 (b'. Archived from the original on ') had 433 occurrences
merge 83/256: (97, 32) -> 338 (b'a ') had 420 occurrences
merge 84/256: (115, 116) -> 339 (b'st') had 409 occurrences
merge 85/256: (105, 99) -> 340 (b'ic') had 406 occurrences
merge 86/256: (46, 91) -> 341 (b'.[') had 381 occurrences
merge 87/256: (101, 99) -> 342 (b'ec') had 374 occurrences
merge 88/256: (105, 301) -> 343 (b'ill') had 367 occurrences
merge 89/256: (39, 262) -> 344 (b"'s ") had 367 occurrences
merge 90/256: (311, 266) -> 345 (b'Taylor Swift ') had 352 occurrences
merge 91/256: (111, 118) -> 346 (b'ov') had 343 occurrences
merge 92/256: (97, 116) -> 347 (b'at') had 334 occurrences
merge 93/256: (97, 262) -> 348 (b'as ') had 315 occurrences
merge 94/256: (101, 262) -> 349 (b'es ') had 309 occurrences
merge 95/256: (74, 117) -> 350 (b'Ju') had 307 occurrences
merge 96/256: (323, 32) -> 351 (b'of ') had 306 occurrences
merge 97/256: (305, 32) -> 352 (b'to ') had 284 occurrences
merge 98/256: (117, 109) -> 353 (b'um') had 281 occurrences
merge 99/256: (84, 331) -> 354 (b'The ') had 277 occurrences
merge 100/256: (271, 100) -> 355 (b'ard') had 277 occurrences
merge 101/256: (263, 32) -> 356 (b'in ') had 276 occurrences
merge 102/256: (270, 32) -> 357 (b'an ') had 276 occurrences
merge 103/256: (101, 108) -> 358 (b'el') had 275 occurrences
merge 104/256: (297, 51) -> 359 (b', 2023') had 271 occurrences
merge 105/256: (271, 273) -> 360 (b'ary ') had 259 occurrences
merge 106/256: (267, 32) -> 361 (b'th ') had 258 occurrences
merge 107/256: (97, 109) -> 362 (b'am') had 257 occurrences
merge 108/256: (108, 273) -> 363 (b'ly ') had 250 occurrences
merge 109/256: (111, 112) -> 364 (b'op') had 249 occurrences
merge 110/256: (311, 116) -> 365 (b'Taylor Swift') had 246 occurrences
merge 111/256: (116, 114) -> 366 (b'tr') had 243 occurrences
merge 112/256: (105, 115) -> 367 (b'is') had 234 occurrences
merge 113/256: (104, 272) -> 368 (b'her ') had 232 occurrences
merge 114/256: (111, 32) -> 369 (b'o ') had 225 occurrences
merge 115/256: (117, 360) -> 370 (b'uary ') had 225 occurrences
merge 116/256: (78, 346) -> 371 (b'Nov') had 222 occurrences
merge 117/256: (312, 340) -> 372 (b'usic') had 221 occurrences
merge 118/256: (371, 314) -> 373 (b'November ') had 221 occurrences
merge 119/256: (101, 119) -> 374 (b'ew') had 219 occurrences
merge 120/256: (97, 266) -> 375 (b'at ') had 219 occurrences
merge 121/256: (108, 32) -> 376 (b'l ') had 218 occurrences
merge 122/256: (58, 32) -> 377 (b': ') had 213 occurrences
merge 123/256: (98, 111) -> 378 (b'bo') had 210 occurrences
merge 124/256: (282, 266) -> 379 (b'Swift ') had 208 occurrences
merge 125/256: (68, 342) -> 380 (b'Dec') had 207 occurrences
merge 126/256: (105, 116) -> 381 (b'it') had 206 occurrences
merge 127/256: (105, 103) -> 382 (b'ig') had 205 occurrences
merge 128/256: (66, 343) -> 383 (b'Bill') had 205 occurrences
merge 129/256: (49, 48) -> 384 (b'10') had 204 occurrences
merge 130/256: (97, 115) -> 385 (b'as') had 203 occurrences
merge 131/256: (264, 103) -> 386 (b'ong') had 202 occurrences
merge 132/256: (79, 99) -> 387 (b'Oc') had 200 occurrences
merge 133/256: (97, 298) -> 388 (b'ati') had 199 occurrences
merge 134/256: (83, 116) -> 389 (b'St') had 198 occurrences
merge 135/256: (387, 305) -> 390 (b'Octo') had 198 occurrences
merge 136/256: (390, 287) -> 391 (b'October ') had 198 occurrences
merge 137/256: (97, 99) -> 392 (b'ac') had 197 occurrences
merge 138/256: (111, 119) -> 393 (b'ow') had 196 occurrences
merge 139/256: (380, 314) -> 394 (b'December ') had 194 occurrences
merge 140/256: (383, 378) -> 395 (b'Billbo') had 191 occurrences
merge 141/256: (97, 100) -> 396 (b'ad') had 190 occurrences
merge 142/256: (108, 101) -> 397 (b'le') had 190 occurrences
merge 143/256: (117, 114) -> 398 (b'ur') had 188 occurrences
merge 144/256: (102, 283) -> 399 (b'for ') had 188 occurrences
merge 145/256: (32, 40) -> 400 (b' (') had 187 occurrences
merge 146/256: (297, 50) -> 401 (b', 2022') had 187 occurrences
merge 147/256: (117, 103) -> 402 (b'ug') had 185 occurrences
merge 148/256: (284, 32) -> 403 (b'ch ') had 184 occurrences
merge 149/256: (115, 266) -> 404 (b'st ') had 181 occurrences
merge 150/256: (321, 110) -> 405 (b'oun') had 176 occurrences
merge 151/256: (98, 353) -> 406 (b'bum') had 172 occurrences
merge 152/256: (111, 108) -> 407 (b'ol') had 171 occurrences
merge 153/256: (312, 266) -> 408 (b'ust ') had 171 occurrences
merge 154/256: (101, 98) -> 409 (b'eb') had 170 occurrences
merge 155/256: (77, 97) -> 410 (b'Ma') had 170 occurrences
merge 156/256: (350, 363) -> 411 (b'July ') had 170 occurrences
merge 157/256: (318, 345) -> 412 (b'). "Taylor Swift ') had 169 occurrences
merge 158/256: (107, 32) -> 413 (b'k ') had 165 occurrences
merge 159/256: (278, 115) -> 414 (b'ers') had 164 occurrences
merge 160/256: (93, 91) -> 415 (b'][') had 164 occurrences
merge 161/256: (65, 402) -> 416 (b'Aug') had 164 occurrences
merge 162/256: (416, 408) -> 417 (b'August ') had 163 occurrences
merge 163/256: (105, 100) -> 418 (b'id') had 161 occurrences
merge 164/256: (297, 49) -> 419 (b', 2021') had 160 occurrences
merge 165/256: (109, 101) -> 420 (b'me') had 159 occurrences
merge 166/256: (101, 112) -> 421 (b'ep') had 156 occurrences
merge 167/256: (261, 49) -> 422 (b'201') had 149 occurrences
merge 168/256: (50, 51) -> 423 (b'23') had 145 occurrences
merge 169/256: (285, 50) -> 424 (b', 2012') had 144 occurrences
merge 170/256: (101, 271) -> 425 (b'ear') had 140 occurrences
merge 171/256: (269, 261) -> 426 (b', 2020') had 140 occurrences
merge 172/256: (73, 110) -> 427 (b'In') had 139 occurrences
merge 173/256: (102, 105) -> 428 (b'fi') had 139 occurrences
merge 174/256: (110, 256) -> 429 (b'ne ') had 139 occurrences
merge 175/256: (395, 355) -> 430 (b'Billboard') had 136 occurrences
merge 176/256: (265, 116) -> 431 (b'rit') had 134 occurrences
merge 177/256: (104, 105) -> 432 (b'hi') had 133 occurrences
merge 178/256: (372, 32) -> 433 (b'usic ') had 133 occurrences
merge 179/256: (304, 34) -> 434 (b'.\n "') had 133 occurrences
merge 180/256: (78, 374) -> 435 (b'New') had 131 occurrences
merge 181/256: (100, 105) -> 436 (b'di') had 130 occurrences
merge 182/256: (65, 112) -> 437 (b'Ap') had 130 occurrences
merge 183/256: (285, 57) -> 438 (b', 2019') had 129 occurrences
merge 184/256: (114, 111) -> 439 (b'ro') had 128 occurrences
merge 185/256: (39, 32) -> 440 (b"' ") had 128 occurrences
merge 186/256: (115, 257) -> 441 (b's, ') had 127 occurrences
merge 187/256: (350, 429) -> 442 (b'June ') had 127 occurrences
merge 188/256: (323, 288) -> 443 (b'of the ') had 126 occurrences
merge 189/256: (99, 291) -> 444 (b'cor') had 126 occurrences
merge 190/256: (50, 49) -> 445 (b'21') had 126 occurrences
merge 191/256: (49, 57) -> 446 (b'19') had 124 occurrences
merge 192/256: (105, 109) -> 447 (b'im') had 123 occurrences
merge 193/256: (290, 32) -> 448 (b'en ') had 123 occurrences
merge 194/256: (409, 114) -> 449 (b'ebr') had 122 occurrences
merge 195/256: (290, 116) -> 450 (b'ent') had 121 occurrences
merge 196/256: (111, 301) -> 451 (b'oll') had 121 occurrences
merge 197/256: (77, 271) -> 452 (b'Mar') had 120 occurrences
merge 198/256: (265, 99) -> 453 (b'ric') had 120 occurrences
merge 199/256: (277, 361) -> 454 (b'with ') had 120 occurrences
merge 200/256: (44, 91) -> 455 (b',[') had 118 occurrences
merge 201/256: (70, 449) -> 456 (b'Febr') had 118 occurrences
merge 202/256: (456, 370) -> 457 (b'February ') had 118 occurrences
merge 203/256: (365, 344) -> 458 (b"Taylor Swift's ") had 118 occurrences
merge 204/256: (300, 430) -> 459 (b'". Billboard') had 118 occurrences
merge 205/256: (101, 97) -> 460 (b'ea') had 116 occurrences
merge 206/256: (285, 54) -> 461 (b', 2016') had 116 occurrences
merge 207/256: (421, 116) -> 462 (b'ept') had 115 occurrences
merge 208/256: (410, 273) -> 463 (b'May ') had 115 occurrences
merge 209/256: (285, 53) -> 464 (b', 2015') had 115 occurrences
merge 210/256: (437, 265) -> 465 (b'Apri') had 115 occurrences
merge 211/256: (465, 376) -> 466 (b'April ') had 115 occurrences
merge 212/256: (108, 256) -> 467 (b'le ') had 113 occurrences
merge 213/256: (65, 119) -> 468 (b'Aw') had 112 occurrences
merge 214/256: (388, 264) -> 469 (b'ation') had 112 occurrences
merge 215/256: (83, 462) -> 470 (b'Sept') had 112 occurrences
merge 216/256: (470, 314) -> 471 (b'September ') had 112 occurrences
merge 217/256: (114, 97) -> 472 (b'ra') had 111 occurrences
merge 218/256: (274, 406) -> 473 (b'album') had 111 occurrences
merge 219/256: (67, 104) -> 474 (b'Ch') had 110 occurrences
merge 220/256: (118, 256) -> 475 (b've ') had 109 occurrences
merge 221/256: (310, 266) -> 476 (b'est ') had 108 occurrences
merge 222/256: (74, 270) -> 477 (b'Jan') had 108 occurrences
merge 223/256: (50, 50) -> 478 (b'22') had 107 occurrences
merge 224/256: (477, 370) -> 479 (b'January ') had 107 occurrences
merge 225/256: (405, 366) -> 480 (b'ountr') had 106 occurrences
merge 226/256: (382, 104) -> 481 (b'igh') had 106 occurrences
merge 227/256: (300, 354) -> 482 (b'". The ') had 106 occurrences
merge 228/256: (359, 304) -> 483 (b', 2023.\n ') had 106 occurrences
merge 229/256: (49, 51) -> 484 (b'13') had 105 occurrences
merge 230/256: (65, 108) -> 485 (b'Al') had 105 occurrences
merge 231/256: (101, 116) -> 486 (b'et') had 105 occurrences
merge 232/256: (310, 115) -> 487 (b'ess') had 103 occurrences
merge 233/256: (452, 403) -> 488 (b'March ') had 103 occurrences
merge 234/256: (117, 116) -> 489 (b'ut') had 102 occurrences
merge 235/256: (119, 431) -> 490 (b'writ') had 101 occurrences
merge 236/256: (108, 111) -> 491 (b'lo') had 99 occurrences
merge 237/256: (115, 386) -> 492 (b'song') had 97 occurrences
merge 238/256: (226, 128) -> 493 (b'\xe2\x80') had 97 occurrences
merge 239/256: (271, 258) -> 494 (b'ard ') had 97 occurrences
merge 240/256: (48, 32) -> 495 (b'0 ') had 97 occurrences
merge 241/256: (117, 108) -> 496 (b'ul') had 96 occurrences
merge 242/256: (50, 52) -> 497 (b'24') had 95 occurrences
merge 243/256: (105, 262) -> 498 (b'is ') had 94 occurrences
merge 244/256: (298, 99) -> 499 (b'tic') had 93 occurrences
merge 245/256: (97, 103) -> 500 (b'ag') had 93 occurrences
merge 246/256: (34, 32) -> 501 (b'" ') had 93 occurrences
merge 247/256: (65, 110) -> 502 (b'An') had 93 occurrences
merge 248/256: (49, 56) -> 503 (b'18') had 93 occurrences
merge 249/256: (102, 291) -> 504 (b'for') had 90 occurrences
merge 250/256: (480, 273) -> 505 (b'ountry ') had 89 occurrences
merge 251/256: (65, 420) -> 506 (b'Ame') had 88 occurrences
merge 252/256: (506, 453) -> 507 (b'Americ') had 88 occurrences
merge 253/256: (32, 84) -> 508 (b' T') had 88 occurrences
merge 254/256: (115, 296) -> 509 (b'sing') had 87 occurrences
merge 255/256: (119, 348) -> 510 (b'was ') had 86 occurrences
merge 256/256: (49, 50) -> 511 (b'12') had 86 occurrences
Training took 13.33 seconds
'''
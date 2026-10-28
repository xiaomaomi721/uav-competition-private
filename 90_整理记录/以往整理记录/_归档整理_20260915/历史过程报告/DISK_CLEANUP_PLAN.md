# Disk cleanup plan - 2026-09-14

## Decision
STOP. **0 files have safe_to_delete=yes; delete nothing.** No deletion, moving, process signals or GPU use.

## Capacity
- Filesystem total: 3,745,119,428,608 bytes.
- Used: 3,543,697,408,000 bytes.
- Available to jhy at snapshot: 11,103,801,344 bytes (10.341 GiB).
- Free including reserved blocks is not available-to-user capacity.
- All 373 ZIPs: 1,658,288,870 logical bytes.
- 31 identity-verified but ineligible ZIPs: 137,282,345 bytes (130.92 MiB).
- Eligible bytes and reclaimed bytes: **0**. 20/25 GiB goal remains unmet; do not relax preservation.

## Identity and closure
Each blocked verified ZIP has exactly 500 prediction members SHA256-identical to retained original PNGs and a preexisting SHA manifest. CANDIDATES.json contains every member name/path/size/SHA plus archive SHA, inode, nlink and retained manifest/source-control SHA. All have nlink=1 and no observed candidate descriptor in readable owned processes; that is not complete closure.
Owned FD visibility was denied. PID1172952 is sd-pam;2142096 subsequently exited. PIDs3803206/4162826 are coordinator-known zombies, not asserted to hold active files. No permission escalation or workaround attempted. All entries remain safe_to_delete=no.

## Preservation
All training directories and checkpoint history preserved: unknown weights; Stage23/Exp23 Golden; Exp47 strong SEMEDA; Stage66 including steps51800/52000/56000 wherever present. Both entire Q2/Q3 trees retained, including historical/current smoke artifacts, logs, manifests and dynamically required visuals. All unconfirmed TTA/cache/logits and diagnostic arrays, unique visuals, datasets and editor/runtime state remain protected or unverified.
Small pinned hashes matched: active GPU2 policy114, active GPU3 policy108, original GPU2 freeze79, corrected smoke2 freeze81. Giant weights intentionally not rehashed; outside-home binary not read. This is not a claim every pinned dependency was fully reverified.

## Execution
No deletion commands or eligible exact-file entries. Main owns any future review/execution. Blocked entries below are not permission to unlink. Any later approved plan must refresh policies/freezes/dynamic smoke receipts; recheck exact owner/device/inode/size/mtime/SHA/nlink, retained-source identities and handles; use individual exact regular files only; measure df and stop when sufficient. No recursive deletion, wildcard or directory removal.

## Exact identity-verified but blocked files

### /home/jhy/exp39_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4525029; allocated upper bound: 4526080
- Archive SHA256: ff5775677ee24e4f4842e2b0ec74219c08035652bd0a66fb0fdda3dbbdf8dfcd
- Retain PNG source: /home/jhy/exp39_submission_standard/predictions
- Retain manifest: /home/jhy/exp39_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 85e69c2aa609af036cd30cad733e51af81494a825066e1358a80188168fac827
- Retain source control: /home/jhy/exp39_submission_standard/run_config.yaml
- Source-control SHA256: 4984a38374a5ed1daf1bd2505f972037c253d8f88af9c7fe9f331ac238364dc6
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/scan_lambda030_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4186040; allocated upper bound: 4186112
- Archive SHA256: 95a599e9e32534e8a23d59ab094f58736e2be7a372e1ed42af56e0c8d9b77e46
- Retain PNG source: /home/jhy/scan_lambda030_submission_standard/predictions
- Retain manifest: /home/jhy/scan_lambda030_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 19a2260275d4ac9566c9aedfaf5426200291498a7266c1f336d0ebd2bd8512dc
- Retain source control: /home/jhy/scan_lambda030_submission_standard/run_config.yaml
- Source-control SHA256: f06677e35c40f555183d3654073e9606aa297e4f129e300918b8682f15116c18
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp40_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4561495; allocated upper bound: 4562944
- Archive SHA256: 503c35d80fe2e4d78d149e32995f0c62878b0b4f26d65bcd9977039d0d1b99d5
- Retain PNG source: /home/jhy/exp40_submission_standard/predictions
- Retain manifest: /home/jhy/exp40_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 8e29ff5f6f7da34ed768b4eb0a6b6790b344807d9de9aacef745d1bb7be498ab
- Retain source control: /home/jhy/exp40_submission_standard/run_config.yaml
- Source-control SHA256: 41d9398d94eee6c09be1b7991deea0151d33071dcdb077e20e776e26effd36fa
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp52_maxentfull_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4438686; allocated upper bound: 4440064
- Archive SHA256: 8a335789444229d236cc36afdb94ac548f0155bd33edf26a155373b66ac9dc68
- Retain PNG source: /home/jhy/exp52_maxentfull_submission_standard/predictions
- Retain manifest: /home/jhy/exp52_maxentfull_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 9c6aa044aa9830c9e8abcd53acadae25f97339df837c527542515285edad32ee
- Retain source control: /home/jhy/exp52_maxentfull_submission_standard/run_config.yaml
- Source-control SHA256: 193e11d9f0f43214fbad38bd09e30718e306516411f206ab210293c3db4bc10a
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp54_sharpen_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4355564; allocated upper bound: 4358144
- Archive SHA256: d0f20ff7aaeed52e22f67072099b719ed63238f3deaf435d76158f923244db0c
- Retain PNG source: /home/jhy/exp54_sharpen_submission_standard/predictions
- Retain manifest: /home/jhy/exp54_sharpen_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 41bd8207fc5fecc1a59a7206deeeaf1cb2d1837619ee76c57c4ebc8cdf6b1548
- Retain source control: /home/jhy/exp54_sharpen_submission_standard/run_config.yaml
- Source-control SHA256: f015b5a1027e2851921df61b1341c2549bf051b8456624ee7499c77f54bc2b96
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp24_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4104486; allocated upper bound: 4108288
- Archive SHA256: cbcac77dbaf1c315b61d3328a3d7ea5464b2fc6f48866b728dda70586ba805f6
- Retain PNG source: /home/jhy/exp24_submission_standard/predictions
- Retain manifest: /home/jhy/exp24_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: b7c4637f8f7ded7f90b848548ebaa8f8de0482316c620a68e393a7651e08d544
- Retain source control: /home/jhy/exp24_submission_standard/run_config.yaml
- Source-control SHA256: 8c399f9b0e59a33af450b8e6d000729ec9b995f02c9ee3845550f027aba7a012
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp50_maxent_l500_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4402164; allocated upper bound: 4403200
- Archive SHA256: e49b4e3c008097e93d3e982431d5ba42c8debe882fcd7aa3d2d85aeaaae45763
- Retain PNG source: /home/jhy/exp50_maxent_l500_submission_standard/predictions
- Retain manifest: /home/jhy/exp50_maxent_l500_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: a1aaee69c5a723815dc3f89c20498a08b699a35f62c6a416a247f96045d234b8
- Retain source control: /home/jhy/exp50_maxent_l500_submission_standard/run_config.yaml
- Source-control SHA256: 92e5c834d6c8a9b13dba7743f398f2c35c9b9b57029be0d49c8cec989d518d81
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp29_submission.zip
- safe_to_delete: **no**; logical bytes: 4601180; allocated upper bound: 4603904
- Archive SHA256: 3848d7a5dfc4e327393d57d8a58ee1905fb2e35cf081a5d2411700f319d1fea0
- Retain PNG source: /home/jhy/exp29_submission/predictions
- Retain manifest: /home/jhy/exp29_submission/manifests/prediction_sha256.csv
- Manifest SHA256: ffdc441f6e73125dbbdc6cfa0988958a19033bca354d194e8dfd4d776f225d76
- Retain source control: /home/jhy/exp29_submission/run_config.yaml
- Source-control SHA256: 4d41e064e0e023d9b9f94d5875065a7deb0d921913d071f4b26c96ac2f6eaf26
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp49_maxent_l100_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4290951; allocated upper bound: 4292608
- Archive SHA256: 2a8bed6938b30f0242dfb53598c593dc722b1a347aa72a64b00dfb14e51eba62
- Retain PNG source: /home/jhy/exp49_maxent_l100_submission_standard/predictions
- Retain manifest: /home/jhy/exp49_maxent_l100_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: f9cbf51c38f17779bea28ac5bcca011130f3223be5e427f2b704349a6b84f224
- Retain source control: /home/jhy/exp49_maxent_l100_submission_standard/run_config.yaml
- Source-control SHA256: f83d54e18a22d684c09275162ac0a0210b5f8ed25f0d6af583d4410f4ea31cfc
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp25_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4545488; allocated upper bound: 4546560
- Archive SHA256: ca97f8fa84be12260e2947cc95fff9e2813b88503f07c6beb62e0757690d5764
- Retain PNG source: /home/jhy/exp25_submission_standard/predictions
- Retain manifest: /home/jhy/exp25_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: b1e095122518a200948692615ff2dc11db7bccba17f13c5643ceefb079fd172f
- Retain source control: /home/jhy/exp25_submission_standard/run_config.yaml
- Source-control SHA256: 46eacd8483963690d986416ef92a142b7fb0886864ba32185debadc0c425540a
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/stage18_submission_exp18_b3_standard.zip
- safe_to_delete: **no**; logical bytes: 4579117; allocated upper bound: 4579328
- Archive SHA256: ea3f0b482370a6bd13f64e5caddff2413ed689d874237d5c07363726b977b77c
- Retain PNG source: /home/jhy/stage18_submission_exp18_b3_standard/predictions
- Retain manifest: /home/jhy/stage18_submission_exp18_b3_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 7a3c8f66da73251d2f38df04e5dd0a0beeefb9f2da0a224ac8cbb8297682e2b8
- Retain source control: /home/jhy/stage18_submission_exp18_b3_standard/run_config.yaml
- Source-control SHA256: ae9a3a873c147f5d1cf9d93c13d0754d0c148b44e7801ee49071d2232b700771
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/scan_alpha020_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4250465; allocated upper bound: 4251648
- Archive SHA256: b566a5ecaef69259e3a8bc7af23dd7bf810f203783afe6dcc0d83ae6f17a0640
- Retain PNG source: /home/jhy/scan_alpha020_submission_standard/predictions
- Retain manifest: /home/jhy/scan_alpha020_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 074da0c18cd198a3074fcec8c010aba1979091f0a1fd647c9d8c618f3c0fedd9
- Retain source control: /home/jhy/scan_alpha020_submission_standard/run_config.yaml
- Source-control SHA256: 396f637a103adef9329c9d35efd463f23613f23c921a0b540d8a6b572d7481ad
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp31_distill_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4715891; allocated upper bound: 4718592
- Archive SHA256: de64444940a99292f9b57b7ae4026a9a6f4795c8bd43f4d81d2c52398c1b3128
- Retain PNG source: /home/jhy/exp31_distill_submission_standard/predictions
- Retain manifest: /home/jhy/exp31_distill_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: a3a858ab920a1c6b53a59d5504f4a8f5e19272eb1decd0e99e9bf8fb91910a7c
- Retain source control: /home/jhy/exp31_distill_submission_standard/run_config.yaml
- Source-control SHA256: d1c633265af656dcbee326ec78a79402e2056af94b7a38075c0dac63cf92ef66
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp26_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4593757; allocated upper bound: 4595712
- Archive SHA256: 27aecafe48dc076f8dcc591a063820e74b83cfabf3be166bcc83ead5935efca9
- Retain PNG source: /home/jhy/exp26_submission_standard/predictions
- Retain manifest: /home/jhy/exp26_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: bfea3a2cad6a3c982dc543d468c41076d75c1d1e64a712b69e236434d2866f37
- Retain source control: /home/jhy/exp26_submission_standard/run_config.yaml
- Source-control SHA256: 149019fa3cfa4753b9a1c860f1325019eebaabfdb4820bcd5789e63bf7414938
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/submit_exp45.zip
- safe_to_delete: **no**; logical bytes: 4493082; allocated upper bound: 4493312
- Archive SHA256: a58c4d73404e47c2a70b5443a5bd9fca16e69a797c4b3746df22ec8a9628feaa
- Retain PNG source: /home/jhy/submit_exp45/predictions
- Retain manifest: /home/jhy/submit_exp45/manifests/prediction_sha256.csv
- Manifest SHA256: c0634079e5001b6d4700e2b817a8ad07e090dbbe1f2798bc6f67b5bf9313f4e5
- Retain source control: /home/jhy/submit_exp45/run_config.yaml
- Source-control SHA256: 9b72f64f4db2cad4f01391064c421a7f8c165552a361b481479953021d6151e8
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/submit_exp46.zip
- safe_to_delete: **no**; logical bytes: 4473412; allocated upper bound: 4476928
- Archive SHA256: 14ee76f4a617819c578fed6f4ee3f49cfd34864f7986e4118b606bb5e5897a1d
- Retain PNG source: /home/jhy/submit_exp46/predictions
- Retain manifest: /home/jhy/submit_exp46/manifests/prediction_sha256.csv
- Manifest SHA256: b87bbd58905012d4c194b83c69bd96d28f9b99ce04dddd2bd3cad7749392e615
- Retain source control: /home/jhy/submit_exp46/run_config.yaml
- Source-control SHA256: d28ef08641649d1e8907ec24489d2fbd2ce76b2078d29bf7cfbcee353dac2857
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp30_submission.zip
- safe_to_delete: **no**; logical bytes: 4611207; allocated upper bound: 4612096
- Archive SHA256: 68f18b6570a63a6c03fb3cd8f2ccbd2308ffe092d4b408cc290641da6aeeebda
- Retain PNG source: /home/jhy/exp30_submission/predictions
- Retain manifest: /home/jhy/exp30_submission/manifests/prediction_sha256.csv
- Manifest SHA256: c87223a842bc120d0df6e643fd895a2530a74c86f32e381bcf0e6c5eb0ed5cda
- Retain source control: /home/jhy/exp30_submission/run_config.yaml
- Source-control SHA256: 328e363fbdf70cde401a0e7a7347c57ff0487065c560072a230f5481bce09f90
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp34_mixstyle_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4386231; allocated upper bound: 4386816
- Archive SHA256: f50c4b0eafd1333a84787a017a31ced9b6a7075392ea685f70499bf6a4b5f799
- Retain PNG source: /home/jhy/exp34_mixstyle_submission_standard/predictions
- Retain manifest: /home/jhy/exp34_mixstyle_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: f92396c81a9fcfc633c14e9230cc544d606a746059c1c5256a0d7387e816161c
- Retain source control: /home/jhy/exp34_mixstyle_submission_standard/run_config.yaml
- Source-control SHA256: b2f4a785f5210cdb0a83cc29df4725581861f4d94246aa712815040f2c6f816b
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/stage55_pseudo_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 3853993; allocated upper bound: 3854336
- Archive SHA256: bf62420d468e0b92e1c3b59941b0ef18f355924c807a37888f02161383c54c2f
- Retain PNG source: /home/jhy/stage55_pseudo_submission_standard/predictions
- Retain manifest: /home/jhy/stage55_pseudo_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 0ab875198d4aea9694bd6c3b567d8aead500ff46b179aba36b5b3817f9c236cc
- Retain source control: /home/jhy/stage55_pseudo_submission_standard/run_config.yaml
- Source-control SHA256: 59bad6ada52a43ba28cf2e45e7c6d68036bc3c1beb992023be17ea3149474a42
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp48_maxent_l010_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4526372; allocated upper bound: 4530176
- Archive SHA256: 190fd50d4da5933187d756e21d0e64c4fa23d1ba88fbb17c336f632074ad3760
- Retain PNG source: /home/jhy/exp48_maxent_l010_submission_standard/predictions
- Retain manifest: /home/jhy/exp48_maxent_l010_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: ee8c21316b313cda78731da1695e2ac259c721a685eb2e0c3a0839dcebb47141
- Retain source control: /home/jhy/exp48_maxent_l010_submission_standard/run_config.yaml
- Source-control SHA256: 3afdbaf4fe7057957cd1376913dd8c0ce86d30766ad932ba3ed9f6583b83aca0
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp21a_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4504849; allocated upper bound: 4505600
- Archive SHA256: 0e3dc8e947d3afe3eb5f1c530db56045f1ffb1488b6bf6da9c36d94d7a2de47d
- Retain PNG source: /home/jhy/exp21a_submission_standard/predictions
- Retain manifest: /home/jhy/exp21a_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 5326999c6878f06850719b21a2e67b06080a4086116cc63ca6342f2bbb522af4
- Retain source control: /home/jhy/exp21a_submission_standard/run_config.yaml
- Source-control SHA256: c0c4b2369bd74efa9c56c7cb2a380b8dc05c93cc93c4a31dfad0355604d5338b
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/submit_exp43.zip
- safe_to_delete: **no**; logical bytes: 4532304; allocated upper bound: 4534272
- Archive SHA256: 83acd9bfcea53b94d061f09ca293dfa86bfc9d4be3a07f8ebabdb94c3b628d1e
- Retain PNG source: /home/jhy/submit_exp43/predictions
- Retain manifest: /home/jhy/submit_exp43/manifests/prediction_sha256.csv
- Manifest SHA256: 7625bf2b8667e36f2b095841485c9d507b9a830ea6d7c48842ded213bb84760d
- Retain source control: /home/jhy/submit_exp43/run_config.yaml
- Source-control SHA256: 178ff6cd4952ee811730e5ceb0719a2f734aa378c1c8704420b14ee08061b8a5
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp27_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4625742; allocated upper bound: 4628480
- Archive SHA256: 66e441128833fbe6116c97c656aff69e6e9a1950acb59c72a33056b21c82bab7
- Retain PNG source: /home/jhy/exp27_submission_standard/predictions
- Retain manifest: /home/jhy/exp27_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 6a2add003425481e5dfb698bad0282159a749ea3f92950eff7e2bf75f734a8de
- Retain source control: /home/jhy/exp27_submission_standard/run_config.yaml
- Source-control SHA256: 04f9724127a93986b9fd4cbc0a064de045f7f7578e26b1c33dd56f2f6f210e6b
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp51_full6796_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4691962; allocated upper bound: 4694016
- Archive SHA256: cd3a18cdbbf8d39ef96c02ecba457678f4b0a19c2cb98510277556be5f69f71b
- Retain PNG source: /home/jhy/exp51_full6796_submission_standard/predictions
- Retain manifest: /home/jhy/exp51_full6796_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: ccf4de5339525457272ef6da5ecce2871c91e09390f8fe3ea408c4b4ead3c290
- Retain source control: /home/jhy/exp51_full6796_submission_standard/run_config.yaml
- Source-control SHA256: 5bcf9ba7e6e309fafe033b163a92b2c40e350155a4e6ffec5608ba24a8f82eff
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/submit_exp44.zip
- safe_to_delete: **no**; logical bytes: 4506385; allocated upper bound: 4509696
- Archive SHA256: 96c8b7163734acbd00e90e9dd6c82eb5866702b5e5e76d5edbe01d0004374a00
- Retain PNG source: /home/jhy/submit_exp44/predictions
- Retain manifest: /home/jhy/submit_exp44/manifests/prediction_sha256.csv
- Manifest SHA256: 603363c54c76202425ba2127b4dd24c36a7313b5b60f8ead806f78419208e7cc
- Retain source control: /home/jhy/submit_exp44/run_config.yaml
- Source-control SHA256: cc6b976701488e0d5126adee259c196b42e619f8cbe961a684663e03837c4287
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp35_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4332260; allocated upper bound: 4333568
- Archive SHA256: b8fc69d536138f2c7cf2977583a1665461242cfe4d2d033bccd97677042eff7d
- Retain PNG source: /home/jhy/exp35_submission_standard/predictions
- Retain manifest: /home/jhy/exp35_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 25c660945ceb5b4701751152ecb1fa9d6c3ed5e0db1a96362198b64953d4d598
- Retain source control: /home/jhy/exp35_submission_standard/run_config.yaml
- Source-control SHA256: 3cd02c8709d76947aaeeb37fbe977103fd9e3ba1a8f5e74fe4edc3e55f2e3751
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/scan_alpha005_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4211815; allocated upper bound: 4214784
- Archive SHA256: 542367eac5a803a33a38ef2602a23e5eab4d338989a4cb530be0eecca02ddb3f
- Retain PNG source: /home/jhy/scan_alpha005_submission_standard/predictions
- Retain manifest: /home/jhy/scan_alpha005_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 4d816b58455f92df428862dfa37f61ad87126516597e8d576669d3ceac94dd1d
- Retain source control: /home/jhy/scan_alpha005_submission_standard/run_config.yaml
- Source-control SHA256: c00bc7917f996852cc4343f0254bb86b28c88f5897948397012540fa809c9c5c
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp33_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4574799; allocated upper bound: 4575232
- Archive SHA256: c0161b98e8e2eb195d3fe8a19c3c7dc76d053a29c6c89f80023ffd48e20039b7
- Retain PNG source: /home/jhy/exp33_submission_standard/predictions
- Retain manifest: /home/jhy/exp33_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 4e739697552ec9a6eb5f17d14d25b38e2feabde0922b65374c75d587ce1184a5
- Retain source control: /home/jhy/exp33_submission_standard/run_config.yaml
- Source-control SHA256: a4496de76955f7c044aa075e335b60618f7ff5b26c5356b7174d502bddb891e4
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/stage22_submission_exp22_b2_standard.zip
- safe_to_delete: **no**; logical bytes: 4118095; allocated upper bound: 4120576
- Archive SHA256: 22ec6dbf6422e4b4db405e8e9cc5c0fae75818b6a9bf41662f74eae51f083115
- Retain PNG source: /home/jhy/stage22_submission_exp22_b2_standard/predictions
- Retain manifest: /home/jhy/stage22_submission_exp22_b2_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 9d4d14f02f4e790c8254149102d24abfa9dd5a1b25f0166e37882845db729713
- Retain source control: /home/jhy/stage22_submission_exp22_b2_standard/run_config.yaml
- Source-control SHA256: a47876f1845e64262746cb62970756f2bbc6287dbb8925d99ebd71b9eb664d24
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/exp28_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4523118; allocated upper bound: 4526080
- Archive SHA256: 6cc457e930d74a2b4e72df364f47271b0afe86c9b12ff4f7e3c1f20a0152bfeb
- Retain PNG source: /home/jhy/exp28_submission_standard/predictions
- Retain manifest: /home/jhy/exp28_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 57fdb914c3eb81702f020af7a63e361e938ca8d4e6aef690f3383d0b58bd2893
- Retain source control: /home/jhy/exp28_submission_standard/run_config.yaml
- Source-control SHA256: d4211b63382b873d439ae50b85eaaa6a68bf3b21a3303d33586f7f9364f14902
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.

### /home/jhy/scan_lambda010_submission_standard.zip
- safe_to_delete: **no**; logical bytes: 4166406; allocated upper bound: 4169728
- Archive SHA256: dc54f47714a475ddcc9c44d20266bdccb4aa343bbafb88443cbecdb542440f4f
- Retain PNG source: /home/jhy/scan_lambda010_submission_standard/predictions
- Retain manifest: /home/jhy/scan_lambda010_submission_standard/manifests/prediction_sha256.csv
- Manifest SHA256: 02efcc0ae129114b62cf91b5935dde1aedb80aa678e925bef51bd3e5d3bb26d3
- Retain source control: /home/jhy/scan_lambda010_submission_standard/run_config.yaml
- Source-control SHA256: 005f79014a18bc236661f8f2736bd4b3e9aeac9dcfe03777d97a468b2cb05931
- Repackage retained PNG member bytes without GPU; ZIP metadata/container SHA need not reproduce. Existing code_snapshot_id may be null; no claim of bitwise inference regeneration.


"""matched_count_glm_null.py restricted to one cortical area (DANDI 000688 electrode-array label).
Usage: python matched_count_glm_null_area.py M1 --session 201503 --check matchednull_M1/201503.json"""
import sys
import single_compartment_null as S
S.AREA = sys.argv.pop(1)
import matched_count_glm_null as MCN
MCN.main()

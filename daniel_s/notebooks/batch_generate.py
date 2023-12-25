import sys
sys.path.append("../../junya/openmm/scripts")

import os
import time
import traceback
from module import function
from module import preprocess
from module import simulation
from openmm.app import PDBFile

# Log capture
# From https://stackoverflow.com/questions/1218933/can-i-redirect-the-stdout-into-some-sort-of-string-buffer
import sys
from io import StringIO

class RedirectOutputs:
    def __init__(self):
        self._stdout = None
        self._stderr = None
        self._string_io = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = sys.stderr = self._string_io = StringIO()
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def __str__(self):
        return self._string_io.getvalue()

def prepare_one(pdbid, data_dir=None):
    if data_dir:
        function.set_data_dir(data_dir)
    if os.path.exists(function.get_data_path(f'{pdbid}/processed/finished.txt')):
        return
    print("Processing", pdbid)

    t0 = time.time()
    ok = True
    with RedirectOutputs() as log:
        try:
            preprocess.prepare_protein(pdbid)
        except Exception as e:
            ok = False
            print(e)
    with open(function.get_data_path(f'{pdbid}/processed/{pdbid}_process.log'),"wb") as f:
        f.write(str(log).encode("utf-8"))
    t1 = time.time() - t0
    finished_str = f"{pdbid} {('error', 'ok')[int(ok)]} ({round(t1,4)} seconds)"
    with open(function.get_data_path(f'{pdbid}/processed/finished.txt'), "w", encoding="utf-8") as finished_file:
        finished_file.write(finished_str)
    print(" ", finished_str)

def simulate_one(pdbid, data_dir=None, steps=10000, report_steps=1):
    if data_dir:
        function.set_data_dir(data_dir)
    finished_file_path = function.get_data_path(f'{pdbid}/simulation/finished.txt')
    if os.path.exists(finished_file_path):
        os.remove(finished_file_path)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("Simulating", pdbid, "on gpu", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Simulating", pdbid)

    t0 = time.time()
    ok = True
    with RedirectOutputs() as log:
        try:
            pdb_path = function.get_data_path(f'{pdbid}/processed/{pdbid}_processed.pdb')
            atom_indices = function.get_non_water_atom_indexes(PDBFile(pdb_path).getTopology())
            simulation.run(pdbid, pdb_path, steps, report_steps=report_steps, atomSubset=atom_indices)
        except Exception as e:
            ok = False
            traceback.print_tb(e.__traceback__)
    with open(function.get_data_path(f'{pdbid}/simulation/{pdbid}_simulation.log'),"wb") as f:
        f.write(str(log).encode("utf-8"))
    t1 = time.time() - t0
    finished_str = f"{pdbid} {('error', 'ok')[int(ok)]} ({round(t1,4)} seconds)"
    with open(finished_file_path, "w", encoding="utf-8") as finished_file:
        finished_file.write(finished_str)
    print(" ", finished_str)

# PDB list for NERSC next 900 (2023.12.20)
# pdbid_list = ['4WCU', '5NSP', '5WI1', '4CMU', '3H0Y', '3S00', '2DS1', '2LL6', '5O0B', '4IJP', '4WT2', '1WAX', '4GK2', '2Q8H', '3SKE', '1UNG', '5NKN', '6HOR', '3MY1', '5C85', '5C28', '3FDN', '3HVG', '2J2I', '4PZW', '2V7A', '6NW3', '3E63', '5OXG', '6G91', '5TX5', '5GGN', '4CS9', '6AJV', '4MGB', '1TQF', '6T1N', '4QR3', '4IBK', '2XL2', '5VA9', '2V85', '4DCE', '4A6B', '3NQ9', '4FAB', '3IVG', '4OBA', '1W84', '1N0S', '6GVF', '3QKM', '2YCM', '4J82', '3Q5U', '3HP9', '2JLE', '3ERK', '3H0J', '5IA3', '4D1B', '2OGY', '3ZRL', '5AZG', '4XH6', '5DHP', '1XWS', '6FGF', '6DRG', '6GLB', '5JAP', '5NHJ', '5K9W', '4DGO', '4L0T', '6UDU', '2FTS', '4Z0U', '2VO5', '2NYR', '2X8I', '5T2L', '3PIZ', '3GBE', '4L7O', '2ZV9', '2R5A', '3HAW', '4GE1', '2J77', '2F4B', '4B72', '5TQ5', '5FDO', '3F8W', '4U79', '4RVT', '3IES', '4YZU', '1T7J', '3OTF', '5Q1E', '2PHB', '4ERZ', '4PD9', '2X9E', '2PIX', '3ZBF', '6AJY', '3TIZ', '4NRB', '1KE5', '5WIN', '2TPI', '2CMA', '6AR4', '4OVH', '4TPM', '4OR4', '4R4T', '1SDV', '2P7A', '5AQQ', '6FT9', '2W8W', '2AVO', '1RQ2', '2ITZ', '1WBG', '3ZN0', '4CY1', '1OGD', '4DTK', '2C6K', '5YC1', '5CF8', '3C3Q', '1QL9', '5NTK', '4JAL', '5Y59', '1V2N', '3PIY', '4XAR', '6MDD', '1S9T', '2YMD', '5H5R', '5AIA', '5G43', '1U3Q', '3B65', '4CIX', '5Q1I', '6QM7', '6U80', '2WMW', '1Y19', '2CNI', '3L08', '3M54', '5EP7', '1U9E', '6BIK', '5YE9', '4Z3V', '1JSV', '3QT6', '5NLK', '5KR0', '3UP7', '3GK4', '2WMV', '6Q4G', '4Y2P', '4AZP', '4HNF', '2KSB', '6CDO', '4RRQ', '4TV3', '4YNK', '6UVY', '5SZB', '6FFE', '1LPG', '1PFY', '4POV', '6SFJ', '3ZQE', '4HVS', '5HNA', '6H9V', '1V2K', '6FT4', '3BE2', '6QYP', '2KGI', '6G37', '3FUI', '4NMX', '4A4V', '5ORB', '3WV2', '4IVD', '4TW6', '5NHY', '4XU1', '3UO5', '5O4F', '5Y62', '2AVQ', '5HMH', '5CAO', '3EYF', '3QA2', '4MY6', '4AZC', '4JR3', '2FGI', '4P90', '2GM9', '5EYZ', '5IZU', '2M0U', '3UOK', '4B0J', '5M7T', '2VJ6', '3KDU', '4LZR', '2WXF', '6AYS', '4M7C', '4STD', '6O5G', '1KUK', '4UVX', '6HGF', '4JQ7', '2R3W', '5UL6', '5MEK', '4EJN', '2CM8', '2R64', '1WCC', '3PSB', '1V0O', '5DDF', '2WOU', '4FHH', '5X9P', '4EH7', '5U6C', '6HOL', '1YKR', '5UOY', '1X7B', '6BMR', '5F6U', '1JD5', '4N4T', '5ANV', '3RV7', '5X54', '6G3Y', '1MEU', '6AXJ', '5K05', '5DLZ', '4KSY', '5U0Y', '1UML', '6TIM', '2AVV', '4BNX', '2P16', '3DP9', '1K4H', '2QI4', '4TUH', '3WIY', '5M56', '3PTY', '1U1B', '5G45', '3BRN', '5ZVW', '6HTN', '5TEG', '6GUC', '1YCM', '1QTI', '4M84', '3TMK', '2JH6', '3HBO', '2PJ8', '5AQN', '3QW7', '4K3Q', '4JBP', '3LJ3', '3DUX', '4IZ0', '4JV6', '1SIV', '4Y85', '4DCS', '5L2M', '1Y6R', '6I14', '1JMF', '1QWF', '4OQ6', '3FZT', '4L09', '3MHW', '4QMW', '2AM2', '4OTF', '5JAT', '5C4S', '5CXH', '6CHQ', '5ZU0', '5WG5', '2XJ2', '6DRT', '2C1P', '1F9G', '3Q32', '3TV7', '1GCA', '5H9S', '5KA3', '5W1E', '2QBR', '4J09', '5YFS', '3QQU', '5KBI', '2I6B', '5AR5', '5JZB', '5AVF', '3UVW', '5VLL', '6HDN', '4ZSH', '2VOT', '6FNF', '5T9U', '2HK5', '6HWU', '4MSU', '2OI3', '2H5E', '4QF7', '6IQL', '4YOZ', '4AGC', '4NBL', '3R42', '4F39', '3N76', '4CNH', '5FQT', '3PE1', '2CNH', '2PMC', '4CR5', '4U5L', '2W6M', '3DPF', '4PSB', '2E1W', '6MQM', '2WTV', '6EGA', '5X4N', '1STP', '1JG0', '2WU6', '1HK2', '5OU2', '2FDP', '3P2E', '2F80', '1NYX', '3D20', '2IEO', '4AT4', '5IH9', '2GNL', '3NTP', '2H2H', '4BIE', '4V25', '5ZUJ', '2NT7', '5ZDC', '5HO8', '1XA5', '4G31', '3PD3', '5Q17', '2IZX', '6HZV', '4NWM', '5CLM', '1N1G', '6AQS', '4MM7', '6PI5', '3P4F', '5I94', '3G90', '4IJ1', '5DYO', '4HDC', '3W9R', '4YV1', '5MKJ', '4OGV', '6SD9', '4B1J', '6FRF', '1ZGI', '3N49', '4AZF', '6G97', '4QRC', '2IQG', '4RWJ', '6GGD', '5K0S', '4UCR', '3RWP', '1LZO', '3MG7', '4JBO', '6DPY', '4OS7', '4AT5', '3KMY', '6NPU', '6GVG', '4JOF', '2ERZ', '6DPT', '4XHL', '3R2Y', '2YEM', '4BGH', '2PEM', '4X63', '6CHM', '4LED', '5Z1T', '4WKC', '4MMP', '4AFT', '1EBY', '4EUO', '3NWW', '6NJI', '3CE3', '5X4M', '4IH6', '5EAK', '5X4P', '4KIJ', '4GU6', '6QGH', '3ZJ8', '4AW5', '2ODD', '1Y2G', '3H9O', '4HIS', '6DUH', '4EQF', '2RA0', '1UU3', '6EFK', '2WXQ', '2AOG', '1NW7', '4ERQ', '1QVT', '6HK3', '2HDQ', '4PY1', '5G46', '3PHE', '3SUS', '6NV7', '3QKD', '2XCK', '2JC0', '6BVH', '6G0W', '3UUG', '3IPU', '4K2G', '1L8G', '5AAB', '4YY6', '4Q4Q', '4LLJ', '5WGD', '4B6R', '4AVW', '1EK1', '1XOQ', '2JO9', '3K5C', '3V31', '2Z7R', '6B5R', '2HZN', '5XS2', '2J95', '1FYR', '4HKI', '5J1R', '5ZYM', '2ITT', '6GEV', '5CP5', '3IG6', '1VKJ', '5NWG', '6CBF', '4UDA', '4WRS', '6CKC', '1VIK', '3RK7', '1FV0', '3TCP', '2AQ9', '4TKH', '4CU8', '3AY0', '1HIM', '1GYX', '5NHO', '1ZEA', '6EVP', '4KBY', '4J46', '1W51', '6DF7', '1HNN', '6MSA', '6MXE', '6RJ2', '5JIY', '2I4Z', '2XM9', '2BTR', '5MGM', '5UN9', '5SZC', '3BT9', '5ENB', '3JWR', '1RS2', '3T2W', '6NPI', '5QAX', '2WKZ', '2JAJ', '5ZK5', '2XYF', '6QR2', '2GEK', '1GFY', '4CWN', '5J47', '2X39', '4RRG', '5NVV', '2O5K', '5X73', '2VTS', '4EW2', '2F35', '2H23', '4Q6R', '2RR4', '3QRK', '4FGY', '2HYY', '2CNE', '5ZQP', '4IBC', '5C4U', '3IPQ', '4TMR', '5NI5', '5IPC', '2G8R', '1FCH', '4XXS', '5WIP', '5HFB', '3VQS', '4D1S', '4EBW', '1XGI', '5I9I', '6A3N', '2ZMM', '1PR1', '2W8J', '5Q1A', '6E4F', '1Q41', '4PUM', '1URC', '3OKI', '2BRP', '2LCT', '4LEQ', '6E99', '4Q93', '2YDK', '4PPA', '5OP4', '4F6W', '4ALG', '3V51', '3NU6', '4B2L', '4W52', '2CTC', '5UEX', '6ET8', '6T1O', '5H5F', '4ZYV', '6DUG', '2YDF', '4YFF', '3M6F', '1XMY', '4HKK', '4JU6', '2CHM', '4F0C', '4I1R', '4BCS', '3MI2', '1UJJ', '3KC0', '2BFQ', '3R7O', '1NHG', '5EGU', '1Q3W', '4DJP', '6MEY', '6J06', '2XYT', '1OHR', '4I10', '5IPJ', '4YLK', '3M93', '2GEJ', '4LQ9', '4O91', '3M67', '5AQO', '6GDG', '1Z6S', '3OCT', '5J7J', '5NK6', '1ZYJ', '4AB9', '1XNX', '3AT4', '4F09', '1F0U', '5KK8', '1XR9', '5C8N', '4BCN', '2JKH', '5T8Q', '3BET', '4ANM', '3Q71', '3LPI', '6O9B', '1OAI', '3ZSX', '5XSR', '2GFS', '3UZJ', '2IN6', '4MEP', '5KLR', '3LQ8', '6OIO', '2AZR', '4KP0', '5BYZ', '4MDR', '4MHA', '5H0H', '5C4K', '1RRY', '5EGS', '3SM0', '5CIN', '3AO1', '4JLN', '5NU1', '6S88', '6I0Z', '4OGJ', '4BDF', '1NNY', '4K9H', '2XJ1', '3KF4', '5I86', '3MWU', '5UEZ', '4L53', '3IVQ', '3MY5', '5Q0W', '2XQQ', '6IM4', '1TSM', '1O0F', '5IA5', '3V2O', '6GJL', '5TR6', '6FR2', '6I3U', '2J62', '5KBG', '4J4N', '1RD4', '3SU4', '6F8G', '2OJ9', '3OHF', '1HTI', '4PV7', '3RWD', '6HY2', '3ST5', '1KAT', '2FCI', '6EJ4', '2WI3', '4H1M', '3V2X', '4BFR', '2XZG', '2JBO', '6Q9U', '6IIK', '3C6W', '3CIB', '4DRK', '5DUS', '5X5O', '1MQJ', '3BXF', '4DRP', '3AVH', '3C4F', '3IFL', '6O6G', '5EOL', '1M51', '4HLM', '1NPW', '6B98', '5HOR', '5FOG', '6K3L', '4UVY', '4QSM', '1YDT', '5FP0', '3C88', '5AQR', '5L2N', '5Q19', '6QMT', '5NIN', '4DUM', '5B5G', '3HO2', '4QYY', '3KQD', '3NOK', '1VJB', '4O9W', '5BNM', '2CGF', '2RCW', '2R02', '6M8B', '3IVH', '2VEU', '2LKK', '2UY3', '6FUH', '3UNK', '2PYM', '6CK3', '6B7C', '5ONE', '4ZYC', '5Q15', '3UMW', '4AXD', '4OD7', '5W4W', '6I3S', '3ARQ', '4A7I', '3QSB', '3DST', '3W0L', '5CGJ', '5ACY', '5OCI', '4DEG', '6AI9', '1S39', '4J0A', '1PGP', '6IMI', '6PG7', '2AOX', '4AUJ', '5JFU', '3RXO', '3E3B', '1NVQ', '5AEN', '4JV9', '1ICJ', '4U44', '5ENH', '4GH6', '5HRV', '6BDY', '1THZ', '3R21', '3LC5', '1K1N', '1SJE', '3SUT', '1YC5', '4Q18', '1ELA', '5X9O', '2VPO', '3FV2', '5ML6', '5K5N', '2RNW', '4LSJ', '5KU9', '3EU7', '6G14', '1Z95', '3PQZ', '5AKI', '5AOJ', '3KGP', '5WF7', '3IMC', '6C4D', '2RG5', '5ZEF', '3K9X', '3EL5', '6GGF', '3RAL', '3FC8', '2AQB', '3WBL', '1PRO', '5W7I']
# PDB list for NERSC next 510 (2023.12.22)
pdbid_list = ['6Q9Q', '4KQR', '2KBS', '5EYM', '6T1L', '4ZJR', '4DFL', '3P7B', '5MRP', '4EB8', '1RTH', '3GUZ', '4JFL', '3AZ8', '5I29', '1LJT', '5TOL', '6GON', '4HGC', '1GSZ', '5NT4', '2OT1', '5ABE', '3CCT', '2P95', '5C7N', '4UIB', '6FZG', '5W0Q', '5O1I', '4CWR', '3MRX', '5T97', '2X7O', '4OMJ', '5ZNL', '3H91', '4W9L', '5T8O', '1MU6', '3B0W', '5K0C', '2QHY', '5NKA', '6PF5', '4JWK', '6HMR', '5HKH', '5UQ9', '4C3K', '3ZM6', '5UFR', '5O55', '4RWL', '6FC6', '5OM3', '1YT9', '6IZQ', '5O1H', '3ZO4', '6MOB', '3V30', '1YET', '3OAP', '2EXC', '5OVF', '1UYM', '4O0R', '1O4F', '3ZI0', '6F1N', '6MXC', '5AIS', '4BKJ', '2EI6', '4QL8', '3RI1', '3Q6K', '4WSY', '2XHT', '2NO3', '5DY7', '3MPT', '2BYI', '5Y94', '5H5Q', '1YYE', '1N9A', '6R9U', '4D1C', '4V24', '3G0W', '4WY7', '4BB4', '3LKH', '2R58', '3AVM', '1XPC', '4QNU', '4K78', '6QTX', '4E35', '5LDO', '6IEZ', '3HNZ', '3PLU', '4MCD', '4PR5', '3D0B', '4KNR', '5TZH', '5AI5', '6CCY', '6MAJ', '1Y6B', '2I0Y', '1UVR', '2ZDX', '2RIN', '2G00', '2XS0', '3FQA', '3RKB', '2GDO', '4DE5', '5YA5', '3LPT', '1XUD', '6MJF', '4E1M', '3K8C', '1JQY', '1G7G', '5T31', '1PXM', '5JAL', '1YWR', '4IBF', '4RLL', '3QW6', '5LY3', '4J7I', '5DJR', '4GS8', '4XAQ', '1LF2', '3UNJ', '1RT9', '2JBP', '6MD0', '1I5R', '5NKC', '4GID', '4E8Z', '1Q4L', '5Z4O', '3QTX', '1FF1', '4DH6', '3PJ2', '5F1V', '4F65', '2XRU', '4PN1', '5KYA', '2QMF', '3SR4', '3KYQ', '3EOU', '2YIX', '2W0Z', '5EH0', '6ORR', '4F14', '1MQD', '1MQH', '5HBJ', '4L23', '1QHC', '2UUO', '1VFN', '3ZYA', '4UIU', '4F7N', '3QLM', '2I5F', '4J3I', '5FE6', '6UDI', '3G6H', '2YXJ', '5KAD', '2WER', '2VJ9', '3UX0', '4BI7', '3POA', '5KZI', '5UFP', '6HZU', '3C4E', '4FXY', '1L2S', '3L3Q', '1JYC', '4A2A', '3T07', '6B1C', '6MNP', '1OWK', '3INH', '1OWJ', '3MO8', '1OBA', '4W9P', '1TOI', '3WB5', '4Q1B', '3MJ1', '4FMO', '4U6X', '4N1U', '3FXV', '5TCI', '3QKK', '3BGM', '3EJ5', '2LSP', '5EM6', '1O46', '4IR5', '3SXF', '4O09', '5W86', '4K4F', '4N98', '5AQT', '6F0Y', '4JU7', '3P4V', '5NT0', '1SOJ', '1OL2', '5HZN', '2XCH', '3ARU', '5H0G', '6NFG', '2QBQ', '3CSO', '4M6P', '4OCQ', '1UYH', '1X1Z', '4C7T', '3MW1', '6BX6', '6HAJ', '4OUE', '5ZK3', '3PRS', '3D5M', '2O4J', '6A04', '4NAU', '5TWL', '1NHW', '1NJA', '3OT3', '4J0V', '5BW4', '5DJP', '2A14', '2B53', '5AJV', '3QQK', '2BRM', '2X4Z', '5JGA', '4N4V', '3HHK', '5NI8', '3OGQ', '4HXL', '5A3X', '4TW7', '4XG7', '6HBN', '4PP5', '1TZE', '2MWP', '2J34', '4XWK', '3EWH', '2ITP', '4RFM', '3OSW', '4JQL', '3OWJ', '3IPE', '2OQI', '5KHI', '5Z4H', '3O9C', '2QE5', '1W1P', '3P3S', '1HP5', '4EM7', '4N9C', '3N51', '4BTT', '3ELC', '3BR9', '1O4B', '2O65', '5NZM', '6PYC', '3WTI', '6AYD', '4ZOM', '3FVH', '3R8I', '1T08', '1TCX', '4J0S', '6I8T', '1SHD', '5AI0', '3ANT', '4F70', '5Q1G', '4F08', '6T6A', '4MZ5', '6QQW', '3QKL', '2PU2', '3AQA', '5XMS', '2C1Q', '5VDW', '2GMK', '2AUC', '4L50', '1P2A', '4AI5', '5J0D', '5Q0O', '4AGN', '5HO7', '1IL9', '4DK5', '4ZE6', '4HDP', '4OEX', '4K55', '4X5Z', '4XJ0', '5F1Z', '5HU1', '4X7I', '2ROX', '2QOH', '3K15', '1NDY', '4G2R', '3ARP', '6QI7', '1W4P', '4HLC', '3BHB', '3IUX', '5F8Y', '2WAJ', '5E0A', '1SDU', '1FJ4', '4JFK', '4JNC', '3VID', '2WI2', '6O50', '4P58', '6IBX', '5AQF', '4ISU', '3GCQ', '4FZ6', '1KUG', '3QI4', '3GY2', '4EDZ', '2VTQ', '1F74', '1W6Y', '1HOS', '6OOY', '1KDK', '2A3A', '2KS9', '4IH5', '6E9W', '1NJJ', '6NPV', '1OTH', '1YY6', '5MOB', '5IJP', '3UH4', '3MVH', '3OEV', '5LAQ', '2G79', '3UDM', '6GJN', '3GI5', '5JYO', '5IAW', '6E6J', '1KZK', '6UVP', '4ACX', '2P4Y', '1WUM', '1V0P', '2IWU', '4RR6', '5K4Z', '3BPR', '1LKL', '2BGE', '1W5W', '3VBW', '4BTK', '5V7W', '3JRX', '6NTB', '5AEP', '4KNE', '4O04', '5AM2', '5O0A', '4ABK', '4RPN', '3LCD', '5EZH', '3WZ8', '3G2Y', '5N9R', '1U0G', '1Y2F', '1JWS', '3KQW', '6GCW', '6FBA', '4BIC', '3RPV', '5Q1H', '3ZMU', '5YZ7', '5O5F', '4X7N', '4BUP', '4O2P', '3NNU', '4URK', '2L1R', '2XEZ', '4NUS', '3BZ3', '1X11', '4MRF', '5B56', '1OIT', '4XU3', '4Q1A', '4UHG', '3UW9', '6QZ6', '4O0Y', '5E4W', '6MLL', '1PZP', '4Q4R', '5T3N', '5AL1', '4BO7', '1HK1', '2CBV', '5NGZ', '5LJ0', '4I8X', '1JP5', '2PWR', '4D85', '3OD0', '4MBL', '3UDQ', '4R5N', '5J2X', '3QAQ', '3RPR', '1RS4', '4GGZ', '3GCS', '4IH3', '1KNE', '3O9D', '5F00']

def init_on_gpu(gpu_list, counter):
    gpu_id = None
    with counter.get_lock():
        gpu_id = gpu_list[counter.value % len(gpu_list)]
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

def main():
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-index", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--pool-size", default=10, type=int, help="Number of simultaneous simulations to run")
    parser.add_argument("--steps", default=10000, type=int, help="Total number of steps to run")
    parser.add_argument("--report-steps", default=1, type=int, help="Save data every n-frames")
    parser.add_argument("--data-dir", default="../data/", type=str)
    parser.add_argument("--gpus", default=None, type=str, help="A comma delimited lists of GPUs to use e.g. '0,1,2,3'")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.data_dir):
        print("Invalid data directory:", args.data_dir)
        return 1

    # if args.batch_size is None or args.batch_index is None:
    #     print("Batch size and index must both be set.")
    #     parser.print_usage()

    try:
        multiprocessing.set_start_method('spawn') # because NERSC says to use this one?
    except Exception as e:
        print("Multiprocessing:", e)

    batch_pdbid_list = pdbid_list[args.batch_index*args.batch_size:(args.batch_index+1)*args.batch_size]
    print(batch_pdbid_list)

    init_function = None
    init_args = None
    if args.gpus is not None:
        gpu_list = [int(i) for i in args.gpus.split(",")]
        init_args = (gpu_list, multiprocessing.Value('i', 0, lock=True))
        init_function = init_on_gpu

    t0 = time.time()
    with multiprocessing.Pool(args.pool_size, initializer=init_function, initargs=init_args) as pool:
        pending_results = []
        for pdbid in batch_pdbid_list:
            pending_results += [pool.apply_async(simulate_one, (pdbid, args.data_dir, args.steps, args.report_steps))]
        
        while pending_results:
            pending_results = [i for i in pending_results if not i.ready()]
            if pending_results:
                pending_results[0].wait(1)
    
    t1 = time.time() - t0
    print(f"Finished {len(batch_pdbid_list)} in {round(t1,4)} seconds")

    return 0

if __name__ == "__main__":
    sys.exit(main())

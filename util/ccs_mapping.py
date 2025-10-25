"""
CCS (Clinical Classifications Software) mapping utilities
Handles mapping between ICD codes and CCS codes
"""
import os
import pandas as pd
import requests
from typing import Dict, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CCSMapper:
    """Class to handle CCS mapping for ICD codes"""
    
    def __init__(self, mapping_file_path: Optional[str] = None):
        """
        Initialize CCS mapper
        
        Args:
            mapping_file_path: Path to local CCS mapping file. If None, will try to download.
        """
        self.icd_to_ccs = {}
        self.ccs_to_icd = {}
        self.mapping_file_path = mapping_file_path
        
        if mapping_file_path and os.path.exists(mapping_file_path):
            self.load_mapping_from_file(mapping_file_path)
        else:
            self.create_default_mapping()
    
    def create_default_mapping(self):
        """
        Create a default mapping based on common ICD-10-CM to CCS mappings
        This is a simplified mapping for demonstration purposes.
        In practice, you should download the official mapping from AHRQ.
        """
        logger.info("Creating default CCS mapping (simplified version)")
        
        # This is a simplified mapping - in practice, you should use the official AHRQ mapping
        default_mappings = {
            # Cardiovascular diseases
            'I10': '98',  # Essential hypertension
            'I11': '98',  # Hypertensive heart disease
            'I12': '98',  # Hypertensive chronic kidney disease
            'I13': '98',  # Hypertensive heart and chronic kidney disease
            'I20': '100', # Angina pectoris
            'I21': '100', # ST elevation myocardial infarction
            'I22': '100', # Subsequent ST elevation myocardial infarction
            'I23': '100', # Certain current complications following ST elevation myocardial infarction
            'I24': '100', # Other acute ischemic heart diseases
            'I25': '100', # Chronic ischemic heart disease
            'I26': '99',  # Pulmonary embolism
            'I27': '99',  # Other pulmonary heart diseases
            'I28': '99',  # Other diseases of pulmonary vessels
            'I30': '99',  # Acute pericarditis
            'I31': '99',  # Other diseases of pericardium
            'I32': '99',  # Pericarditis in diseases classified elsewhere
            'I33': '99',  # Acute and subacute endocarditis
            'I34': '99',  # Nonrheumatic mitral valve disorders
            'I35': '99',  # Nonrheumatic aortic valve disorders
            'I36': '99',  # Nonrheumatic tricuspid valve disorders
            'I37': '99',  # Nonrheumatic pulmonary valve disorders
            'I38': '99',  # Endocarditis, valve unspecified
            'I39': '99',  # Endocarditis and heart valve disorders in diseases classified elsewhere
            'I40': '99',  # Acute myocarditis
            'I41': '99',  # Myocarditis in diseases classified elsewhere
            'I42': '99',  # Cardiomyopathy
            'I43': '99',  # Cardiomyopathy in diseases classified elsewhere
            'I44': '99',  # Atrioventricular and left bundle-branch block
            'I45': '99',  # Other conduction disorders
            'I46': '99',  # Cardiac arrest
            'I47': '99',  # Paroxysmal tachycardia
            'I48': '99',  # Atrial fibrillation and flutter
            'I49': '99',  # Other cardiac arrhythmias
            'I50': '99',  # Heart failure
            'I51': '99',  # Ill-defined descriptions and complications of heart disease
            'I52': '99',  # Other heart disorders in diseases classified elsewhere
            'I60': '99',  # Nontraumatic subarachnoid hemorrhage
            'I61': '99',  # Nontraumatic intracerebral hemorrhage
            'I62': '99',  # Other and unspecified nontraumatic intracranial hemorrhage
            'I63': '99',  # Cerebral infarction
            'I64': '99',  # Stroke, not specified as hemorrhage or infarction
            'I65': '99',  # Occlusion and stenosis of precerebral arteries, not resulting in cerebral infarction
            'I66': '99',  # Occlusion and stenosis of cerebral arteries, not resulting in cerebral infarction
            'I67': '99',  # Other cerebrovascular diseases
            'I68': '99',  # Cerebrovascular disorders in diseases classified elsewhere
            'I69': '99',  # Sequelae of cerebrovascular disease
            'I70': '99',  # Atherosclerosis
            'I71': '99',  # Aortic aneurysm and dissection
            'I72': '99',  # Other aneurysm and dissection
            'I73': '99',  # Other peripheral vascular diseases
            'I74': '99',  # Arterial embolism and thrombosis
            'I77': '99',  # Arteriovenous and lymphatic disorders
            'I78': '99',  # Diseases of capillaries
            'I79': '99',  # Disorders of arteries, arterioles and capillaries in diseases classified elsewhere
            'I80': '99',  # Phlebitis and thrombophlebitis
            'I81': '99',  # Portal vein thrombosis
            'I82': '99',  # Other venous embolism and thrombosis
            'I83': '99',  # Varicose veins of lower extremities
            'I84': '99',  # Hemorrhoids
            'I85': '99',  # Esophageal varices
            'I86': '99',  # Varicose veins of other sites
            'I87': '99',  # Other disorders of veins
            'I88': '99',  # Nonspecific lymphadenitis
            'I89': '99',  # Other noninfective disorders of lymphatic vessels and lymph nodes
            'I95': '99',  # Hypotension
            'I97': '99',  # Intraoperative and postprocedural complications and disorders of circulatory system, not elsewhere classified
            'I99': '99',  # Other and unspecified disorders of circulatory system
            
            # Diabetes
            'E08': '50',  # Diabetes mellitus due to underlying condition
            'E09': '50',  # Drug or chemical induced diabetes mellitus
            'E10': '50',  # Type 1 diabetes mellitus
            'E11': '50',  # Type 2 diabetes mellitus
            'E13': '50',  # Other specified diabetes mellitus
            
            # Respiratory diseases
            'J40': '127', # Bronchitis, not specified as acute or chronic
            'J41': '127', # Simple and mucopurulent chronic bronchitis
            'J42': '127', # Unspecified chronic bronchitis
            'J43': '127', # Emphysema
            'J44': '127', # Other chronic obstructive pulmonary disease
            'J45': '127', # Asthma
            'J46': '127', # Status asthmaticus
            'J47': '127', # Bronchiectasis
            'J60': '127', # Coalworker pneumoconiosis
            'J61': '127', # Pneumoconiosis due to asbestos and other mineral fibers
            'J62': '127', # Pneumoconiosis due to dust containing silica
            'J63': '127', # Pneumoconiosis due to other inorganic dusts
            'J64': '127', # Unspecified pneumoconiosis
            'J65': '127', # Pneumoconiosis associated with tuberculosis
            'J66': '127', # Airway disease due to specific organic dusts
            'J67': '127', # Hypersensitivity pneumonitis due to organic dust
            'J68': '127', # Respiratory conditions due to inhalation of chemicals, gases, fumes and vapors
            'J69': '127', # Pneumonitis due to solids and liquids
            'J70': '127', # Respiratory conditions due to other external agents
            'J80': '127', # Acute respiratory distress syndrome
            'J81': '127', # Pulmonary edema
            'J82': '127', # Pulmonary eosinophilia
            'J84': '127', # Other interstitial pulmonary diseases
            'J85': '127', # Abscess of lung and mediastinum
            'J86': '127', # Pyothorax
            'J90': '127', # Pleural effusion, not elsewhere classified
            'J91': '127', # Pleural effusion in conditions classified elsewhere
            'J92': '127', # Pleural plaque
            'J93': '127', # Pneumothorax
            'J94': '127', # Other pleural conditions
            'J95': '127', # Intraoperative and postprocedural complications and disorders of respiratory system, not elsewhere classified
            'J96': '127', # Respiratory failure, not elsewhere classified
            'J98': '127', # Other respiratory disorders
            'J99': '127', # Respiratory disorders in diseases classified elsewhere
            
            # Kidney diseases
            'N17': '157', # Acute kidney failure
            'N18': '157', # Chronic kidney disease
            'N19': '157', # Unspecified kidney failure
            'N25': '157', # Disorders resulting from impaired renal function
            'N26': '157', # Unspecified contracted kidney
            'N27': '157', # Small kidney of unknown cause
            'N28': '157', # Other disorders of kidney and ureter, not elsewhere classified
            'N29': '157', # Other disorders of kidney and ureter in diseases classified elsewhere
            'N30': '157', # Cystitis
            'N31': '157', # Neuromuscular dysfunction of bladder, not elsewhere classified
            'N32': '157', # Other disorders of bladder
            'N33': '157', # Bladder disorders in diseases classified elsewhere
            'N34': '157', # Urethritis and urethral syndrome
            'N35': '157', # Urethral stricture
            'N36': '157', # Other disorders of urethra
            'N37': '157', # Urethral disorders in diseases classified elsewhere
            'N39': '157', # Other disorders of urinary system
            'N40': '157', # Benign prostatic hyperplasia
            'N41': '157', # Inflammatory diseases of prostate
            'N42': '157', # Other and unspecified disorders of prostate
            'N43': '157', # Hydrocele and spermatocele
            'N44': '157', # Torsion of testis
            'N45': '157', # Orchitis and epididymitis
            'N46': '157', # Male infertility
            'N47': '157', # Disorders of prepuce
            'N48': '157', # Other disorders of penis
            'N49': '157', # Inflammatory disorders of male genital organs, not elsewhere classified
            'N50': '157', # Other and unspecified disorders of male genital organs
            'N51': '157', # Disorders of male genital organs in diseases classified elsewhere
            'N60': '157', # Benign mammary dysplasia
            'N61': '157', # Inflammatory disorders of breast
            'N62': '157', # Hypertrophy of breast
            'N63': '157', # Unspecified lump in breast
            'N64': '157', # Other disorders of breast
            'N65': '157', # Deformity and disproportion of reconstructed breast
            'N70': '157', # Salpingitis and oophoritis
            'N71': '157', # Inflammatory disease of uterus, except cervix
            'N72': '157', # Inflammatory disease of cervix uteri
            'N73': '157', # Other female pelvic inflammatory diseases
            'N74': '157', # Female pelvic inflammatory disorders in diseases classified elsewhere
            'N75': '157', # Diseases of Bartholin gland
            'N76': '157', # Other inflammation of vagina and vulva
            'N77': '157', # Vulvovaginal ulceration and inflammation in diseases classified elsewhere
            'N80': '157', # Endometriosis
            'N81': '157', # Female genital prolapse
            'N82': '157', # Fistulae involving female genital tract
            'N83': '157', # Noninflammatory disorders of ovary, fallopian tube and broad ligament
            'N84': '157', # Polyp of female genital tract
            'N85': '157', # Other noninflammatory disorders of uterus, except cervix
            'N86': '157', # Erosion and ectropion of cervix uteri
            'N87': '157', # Dysplasia of cervix uteri
            'N88': '157', # Other noninflammatory disorders of cervix uteri
            'N89': '157', # Other noninflammatory disorders of vagina
            'N90': '157', # Other noninflammatory disorders of vulva and perineum
            'N91': '157', # Absent, scanty and rare menstruation
            'N92': '157', # Excessive, frequent and irregular menstruation
            'N93': '157', # Other abnormal uterine and vaginal bleeding
            'N94': '157', # Pain and other conditions associated with female genital organs and menstrual cycle
            'N95': '157', # Menopausal and perimenopausal disorders
            'N96': '157', # Habitual aborter
            'N97': '157', # Female infertility
            'N98': '157', # Complications associated with artificial fertilization
            'N99': '157', # Intraoperative and postprocedural complications and disorders of genitourinary system, not elsewhere classified
            
            # Cancer
            'C00': '11',  # Malignant neoplasm of lip
            'C01': '11',  # Malignant neoplasm of base of tongue
            'C02': '11',  # Malignant neoplasm of other and unspecified parts of tongue
            'C03': '11',  # Malignant neoplasm of gum
            'C04': '11',  # Malignant neoplasm of floor of mouth
            'C05': '11',  # Malignant neoplasm of palate
            'C06': '11',  # Malignant neoplasm of other and unspecified parts of mouth
            'C07': '11',  # Malignant neoplasm of parotid gland
            'C08': '11',  # Malignant neoplasm of other and unspecified major salivary glands
            'C09': '11',  # Malignant neoplasm of tonsil
            'C10': '11',  # Malignant neoplasm of oropharynx
            'C11': '11',  # Malignant neoplasm of nasopharynx
            'C12': '11',  # Malignant neoplasm of pyriform sinus
            'C13': '11',  # Malignant neoplasm of hypopharynx
            'C14': '11',  # Malignant neoplasm of other and ill-defined sites in the lip, oral cavity and pharynx
            'C15': '11',  # Malignant neoplasm of esophagus
            'C16': '11',  # Malignant neoplasm of stomach
            'C17': '11',  # Malignant neoplasm of small intestine
            'C18': '11',  # Malignant neoplasm of colon
            'C19': '11',  # Malignant neoplasm of rectosigmoid junction
            'C20': '11',  # Malignant neoplasm of rectum
            'C21': '11',  # Malignant neoplasm of anus and anal canal
            'C22': '11',  # Malignant neoplasm of liver and intrahepatic bile ducts
            'C23': '11',  # Malignant neoplasm of gallbladder
            'C24': '11',  # Malignant neoplasm of other and unspecified parts of biliary tract
            'C25': '11',  # Malignant neoplasm of pancreas
            'C26': '11',  # Malignant neoplasm of other and ill-defined digestive organs
            'C30': '11',  # Malignant neoplasm of nasal cavity and middle ear
            'C31': '11',  # Malignant neoplasm of accessory sinuses
            'C32': '11',  # Malignant neoplasm of larynx
            'C33': '11',  # Malignant neoplasm of trachea
            'C34': '11',  # Malignant neoplasm of bronchus and lung
            'C37': '11',  # Malignant neoplasm of thymus
            'C38': '11',  # Malignant neoplasm of heart, mediastinum and pleura
            'C39': '11',  # Malignant neoplasm of other and ill-defined sites in the respiratory system and intrathoracic organs
            'C40': '11',  # Malignant neoplasm of bone and articular cartilage of limbs
            'C41': '11',  # Malignant neoplasm of bone and articular cartilage of other and unspecified sites
            'C43': '11',  # Malignant melanoma of skin
            'C44': '11',  # Other malignant neoplasms of skin
            'C45': '11',  # Mesothelioma
            'C46': '11',  # Kaposi sarcoma
            'C47': '11',  # Malignant neoplasm of peripheral nerves and autonomic nervous system
            'C48': '11',  # Malignant neoplasm of retroperitoneum and peritoneum
            'C49': '11',  # Malignant neoplasm of other connective and soft tissue
            'C50': '11',  # Malignant neoplasm of breast
            'C51': '11',  # Malignant neoplasm of vulva
            'C52': '11',  # Malignant neoplasm of vagina
            'C53': '11',  # Malignant neoplasm of cervix uteri
            'C54': '11',  # Malignant neoplasm of corpus uteri
            'C55': '11',  # Malignant neoplasm of uterus, part unspecified
            'C56': '11',  # Malignant neoplasm of ovary
            'C57': '11',  # Malignant neoplasm of other and unspecified female genital organs
            'C58': '11',  # Malignant neoplasm of placenta
            'C60': '11',  # Malignant neoplasm of penis
            'C61': '11',  # Malignant neoplasm of prostate
            'C62': '11',  # Malignant neoplasm of testis
            'C63': '11',  # Malignant neoplasm of other and unspecified male genital organs
            'C64': '11',  # Malignant neoplasm of kidney, except renal pelvis
            'C65': '11',  # Malignant neoplasm of renal pelvis
            'C66': '11',  # Malignant neoplasm of ureter
            'C67': '11',  # Malignant neoplasm of bladder
            'C68': '11',  # Malignant neoplasm of other and unspecified urinary organs
            'C69': '11',  # Malignant neoplasm of eye and adnexa
            'C70': '11',  # Malignant neoplasm of meninges
            'C71': '11',  # Malignant neoplasm of brain
            'C72': '11',  # Malignant neoplasm of spinal cord, cranial nerves and other parts of central nervous system
            'C73': '11',  # Malignant neoplasm of thyroid gland
            'C74': '11',  # Malignant neoplasm of adrenal gland
            'C75': '11',  # Malignant neoplasm of other endocrine glands and related structures
            'C76': '11',  # Malignant neoplasm of other and ill-defined sites
            'C77': '11',  # Secondary and unspecified malignant neoplasm of lymph nodes
            'C78': '11',  # Secondary malignant neoplasm of respiratory and digestive organs
            'C79': '11',  # Secondary malignant neoplasm of other and unspecified sites
            'C80': '11',  # Malignant neoplasm without specification of site
            'C81': '11',  # Hodgkin lymphoma
            'C82': '11',  # Follicular lymphoma
            'C83': '11',  # Non-follicular lymphoma
            'C84': '11',  # Mature T/NK-cell lymphomas
            'C85': '11',  # Other and unspecified types of non-Hodgkin lymphoma
            'C86': '11',  # Other specified types of T/NK-cell lymphoma
            'C88': '11',  # Malignant immunoproliferative diseases and certain other B-cell lymphomas
            'C90': '11',  # Multiple myeloma and malignant plasma cell neoplasms
            'C91': '11',  # Lymphoid leukemia
            'C92': '11',  # Myeloid leukemia
            'C93': '11',  # Monocytic leukemia
            'C94': '11',  # Other leukemias of specified cell type
            'C95': '11',  # Leukemia of unspecified cell type
            'C96': '11',  # Other and unspecified malignant neoplasms of lymphoid, hematopoietic and related tissue
            
            # Mental health
            'F01': '653', # Vascular dementia
            'F02': '653', # Dementia in other diseases classified elsewhere
            'F03': '653', # Unspecified dementia
            'F04': '653', # Amnestic disorder due to known physiological condition
            'F05': '653', # Delirium due to known physiological condition
            'F06': '653', # Other mental disorders due to known physiological condition
            'F07': '653', # Personality and behavioral disorders due to known physiological condition
            'F09': '653', # Unspecified mental disorder due to known physiological condition
            'F10': '653', # Alcohol-related disorders
            'F11': '653', # Opioid-related disorders
            'F12': '653', # Cannabis-related disorders
            'F13': '653', # Sedative, hypnotic or anxiolytic-related disorders
            'F14': '653', # Cocaine-related disorders
            'F15': '653', # Other stimulant-related disorders
            'F16': '653', # Hallucinogen-related disorders
            'F17': '653', # Nicotine dependence
            'F18': '653', # Inhalant-related disorders
            'F19': '653', # Other psychoactive substance-related disorders
            'F20': '653', # Schizophrenia
            'F21': '653', # Schizotypal disorder
            'F22': '653', # Delusional disorders
            'F23': '653', # Brief psychotic disorder
            'F24': '653', # Schizoaffective disorders
            'F25': '653', # Schizoaffective disorders
            'F28': '653', # Other psychotic disorders not due to a substance or known physiological condition
            'F29': '653', # Unspecified psychosis not due to a substance or known physiological condition
            'F30': '653', # Manic episode
            'F31': '653', # Bipolar disorder
            'F32': '653', # Major depressive disorder, single episode
            'F33': '653', # Major depressive disorder, recurrent
            'F34': '653', # Persistent mood disorders
            'F39': '653', # Unspecified mood disorder
            'F40': '653', # Phobic anxiety disorders
            'F41': '653', # Other anxiety disorders
            'F42': '653', # Obsessive-compulsive disorder
            'F43': '653', # Reaction to severe stress, and adjustment disorders
            'F44': '653', # Dissociative disorders
            'F45': '653', # Somatoform disorders
            'F48': '653', # Other neurotic disorders
            'F50': '653', # Eating disorders
            'F51': '653', # Nonorganic sleep disorders
            'F52': '653', # Sexual dysfunction, not caused by organic disorder or disease
            'F53': '653', # Mental and behavioral disorders associated with the puerperium, not elsewhere classified
            'F54': '653', # Psychological and behavioral factors associated with disorders or diseases classified elsewhere
            'F55': '653', # Abuse of non-psychoactive substances
            'F59': '653', # Unspecified behavioral syndromes associated with physiological disturbances and physical factors
            'F60': '653', # Personality disorders
            'F63': '653', # Impulse disorders
            'F64': '653', # Gender identity disorders
            'F65': '653', # Paraphilias
            'F66': '653', # Psychological and behavioral disorders associated with sexual development and orientation
            'F68': '653', # Other disorders of adult personality and behavior
            'F69': '653', # Unspecified disorder of adult personality and behavior
            'F70': '653', # Mild intellectual disabilities
            'F71': '653', # Moderate intellectual disabilities
            'F72': '653', # Severe intellectual disabilities
            'F73': '653', # Profound intellectual disabilities
            'F78': '653', # Other intellectual disabilities
            'F79': '653', # Unspecified intellectual disabilities
            'F80': '653', # Specific developmental disorders of speech and language
            'F81': '653', # Specific developmental disorders of scholastic skills
            'F82': '653', # Specific developmental disorder of motor function
            'F84': '653', # Pervasive developmental disorders
            'F88': '653', # Other disorders of psychological development
            'F89': '653', # Unspecified disorder of psychological development
            'F90': '653', # Attention-deficit hyperactivity disorders
            'F91': '653', # Conduct disorders
            'F93': '653', # Emotional disorders with onset specific to childhood
            'F94': '653', # Disorders of social functioning with onset specific to childhood and adolescence
            'F95': '653', # Tic disorders
            'F98': '653', # Other behavioral and emotional disorders with onset usually occurring in childhood and adolescence
            'F99': '653', # Unspecified mental disorder
            
            # Add more mappings as needed...
        }
        
        self.icd_to_ccs = default_mappings
        self.ccs_to_icd = {ccs: [icd for icd, ccs_code in default_mappings.items() if ccs_code == ccs] 
                          for ccs in set(default_mappings.values())}
        
        logger.info(f"Created default mapping with {len(self.icd_to_ccs)} ICD codes")
    
    def load_mapping_from_file(self, file_path: str):
        """
        Load CCS mapping from file
        
        Args:
            file_path: Path to the mapping file (CSV or Excel)
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Assume columns are 'ICD_CODE' and 'CCS_CODE' or similar
            icd_col = None
            ccs_col = None
            
            for col in df.columns:
                if 'icd' in col.lower():
                    icd_col = col
                elif 'ccs' in col.lower():
                    ccs_col = col
            
            if icd_col is None or ccs_col is None:
                raise ValueError(f"Could not find ICD and CCS columns in {file_path}")
            
            # Create mapping dictionaries
            self.icd_to_ccs = dict(zip(df[icd_col], df[ccs_col]))
            self.ccs_to_icd = {}
            
            for icd, ccs in self.icd_to_ccs.items():
                if ccs not in self.ccs_to_icd:
                    self.ccs_to_icd[ccs] = []
                self.ccs_to_icd[ccs].append(icd)
            
            logger.info(f"Loaded mapping from {file_path}: {len(self.icd_to_ccs)} ICD codes")
            
        except Exception as e:
            logger.error(f"Error loading mapping from {file_path}: {e}")
            logger.info("Falling back to default mapping")
            self.create_default_mapping()
    
    def get_ccs_code(self, icd_code: str) -> Optional[str]:
        """
        Get CCS code for an ICD code
        
        Args:
            icd_code: ICD code
            
        Returns:
            CCS code or None if not found
        """
        return self.icd_to_ccs.get(icd_code)
    
    def get_icd_codes(self, ccs_code: str) -> List[str]:
        """
        Get all ICD codes for a CCS code
        
        Args:
            ccs_code: CCS code
            
        Returns:
            List of ICD codes
        """
        return self.ccs_to_icd.get(ccs_code, [])
    
    def map_icd_to_ccs(self, icd_codes: List[str]) -> List[str]:
        """
        Map a list of ICD codes to CCS codes
        
        Args:
            icd_codes: List of ICD codes
            
        Returns:
            List of CCS codes (excluding None values)
        """
        ccs_codes = []
        for icd in icd_codes:
            ccs = self.get_ccs_code(icd)
            if ccs is not None:
                ccs_codes.append(ccs)
        return ccs_codes
    
    def get_mapping_stats(self) -> Dict:
        """
        Get statistics about the mapping
        
        Returns:
            Dictionary with mapping statistics
        """
        return {
            'total_icd_codes': len(self.icd_to_ccs),
            'total_ccs_codes': len(self.ccs_to_icd),
            'avg_icd_per_ccs': sum(len(icds) for icds in self.ccs_to_icd.values()) / len(self.ccs_to_icd) if self.ccs_to_icd else 0
        }


def download_ccs_mapping(url: str = None, output_path: str = "ccs_mapping.csv") -> bool:
    """
    Download CCS mapping from AHRQ website
    
    Args:
        url: URL to download from (if None, uses default AHRQ URL)
        output_path: Path to save the downloaded file
        
    Returns:
        True if successful, False otherwise
    """
    if url is None:
        # Default AHRQ CCS mapping URL (this may need to be updated)
        url = "https://hcup-us.ahrq.gov/toolssoftware/ccs10/ccs10.jsp"
    
    try:
        logger.info(f"Attempting to download CCS mapping from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded CCS mapping to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download CCS mapping: {e}")
        return False


# Global mapper instance
_ccs_mapper = None

def get_ccs_mapper(mapping_file_path: Optional[str] = None) -> CCSMapper:
    """
    Get global CCS mapper instance
    
    Args:
        mapping_file_path: Path to mapping file (only used on first call)
        
    Returns:
        CCSMapper instance
    """
    global _ccs_mapper
    if _ccs_mapper is None:
        _ccs_mapper = CCSMapper(mapping_file_path)
    return _ccs_mapper

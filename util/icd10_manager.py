"""
ICD-10-CM codes management module
Reads and manages the complete ICD-10-CM code set for embedding training
"""
import os
import logging
from typing import List, Set, Dict, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ICD10CodeManager:
    """Manages ICD-10-CM codes reading and loading"""
    
    def __init__(self, data_dir: str = "~/data/MIMIC_IV"):
        """
        Initialize ICD-10 code manager
        
        Args:
            data_dir: Directory containing ICD-10-CM files
        """
        self.data_dir = Path(data_dir).expanduser()
        
        # File paths for ICD-10-CM codes
        self.icd10_files = {
            "2024": {
                "codes": "icd10cm-codes-April-2024.txt",
                "order": "icd10cm-order-April-2024.txt"
            }
        }
        
        self.all_codes: Set[str] = set()
        self.code_descriptions: Dict[str, str] = {}
    
    def check_icd10_files(self, year: str = "2024") -> bool:
        """
        Check if ICD-10-CM files exist locally
        
        Args:
            year: Year of ICD-10-CM codes to check
            
        Returns:
            True if files exist, False otherwise
        """
        if year not in self.icd10_files:
            logger.error(f"Unsupported year: {year}. Available years: {list(self.icd10_files.keys())}")
            return False
        
        files_info = self.icd10_files[year]
        codes_file = self.data_dir / files_info["codes"]
        order_file = self.data_dir / files_info["order"]
        
        if not codes_file.exists():
            logger.error(f"ICD-10-CM codes file not found: {codes_file}")
            return False
            
        if not order_file.exists():
            logger.error(f"ICD-10-CM order file not found: {order_file}")
            return False
        
        logger.info(f"Found ICD-10-CM files for year {year}")
        logger.info(f"Codes file: {codes_file}")
        logger.info(f"Order file: {order_file}")
        return True
    
    def parse_icd10_files(self, year: str = "2024") -> bool:
        """
        Parse ICD-10-CM files and extract codes
        
        Args:
            year: Year of ICD-10-CM codes to parse
            
        Returns:
            True if successful, False otherwise
        """
        if year not in self.icd10_files:
            logger.error(f"Unsupported year: {year}. Available years: {list(self.icd10_files.keys())}")
            return False
        
        files_info = self.icd10_files[year]
        codes_file = self.data_dir / files_info["codes"]
        order_file = self.data_dir / files_info["order"]
        
        if not codes_file.exists() or not order_file.exists():
            logger.error(f"ICD-10-CM files not found for year {year}")
            return False
        
        try:
            logger.info(f"Parsing ICD-10-CM codes from {codes_file}")
            
            codes = set()
            descriptions = {}
            
            # Parse the codes file (format: CODE    DESCRIPTION)
            with open(codes_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Parse format: CODE DESCRIPTION (variable spaces separator)
                    # Try different space patterns to separate code and description
                    import re
                    
                    # First try 3+ spaces (for shorter codes)
                    parts = re.split(r'\s{3,}', line, 1)
                    if len(parts) < 2:
                        # If that fails, try 1+ spaces (for longer codes)
                        parts = re.split(r'\s+', line, 1)
                    
                    if len(parts) >= 2:
                        code = parts[0].strip()
                        description = parts[1].strip()
                        
                        # Validate ICD-10-CM code format
                        if self._is_valid_icd10_code(code):
                            codes.add(code)
                            descriptions[code] = description
            
            # Also parse the order file for additional codes
            logger.info(f"Parsing ICD-10-CM order file from {order_file}")
            with open(order_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Parse format: ORDER_CODE CODE LEVEL DESCRIPTION SHORT_DESCRIPTION
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        code = parts[1].strip()
                        description = parts[3].strip()
                        
                        # Validate ICD-10-CM code format
                        if self._is_valid_icd10_code(code):
                            codes.add(code)
                            if code not in descriptions:  # Don't overwrite if already exists
                                descriptions[code] = description
            
            self.all_codes = codes
            self.code_descriptions = descriptions
            
            logger.info(f"Parsed {len(codes)} ICD-10-CM codes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to parse ICD-10-CM files: {e}")
            return False
    
    def _is_valid_icd10_code(self, code: str) -> bool:
        """
        Validate ICD-10-CM code format
        
        Args:
            code: Code to validate
            
        Returns:
            True if valid ICD-10-CM code, False otherwise
        """
        if not code:
            return False
        
        # ICD-10-CM format: Letter followed by alphanumeric characters
        # Examples: A09, A33, A000, A0100, S52601D, C4A0, C4A10, M1A00X0, Z990, etc.
        import re
        pattern = r'^[A-Z][A-Z0-9]*$'
        return bool(re.match(pattern, code))
    
    def get_all_codes(self) -> List[str]:
        """
        Get all ICD-10-CM codes
        
        Returns:
            List of all ICD-10-CM codes
        """
        return sorted(list(self.all_codes))
    
    def get_code_description(self, code: str) -> Optional[str]:
        """
        Get description for a specific code
        
        Args:
            code: ICD-10-CM code
            
        Returns:
            Description if found, None otherwise
        """
        return self.code_descriptions.get(code)
    
    def create_comprehensive_code_set(self, year: str = "2024") -> List[str]:
        """
        Create comprehensive ICD-10-CM code set by reading local files
        
        Args:
            year: Year of ICD-10-CM codes
            
        Returns:
            List of all ICD-10-CM codes
        """
        # Check if files exist
        if not self.check_icd10_files(year):
            logger.warning("Failed to find ICD-10-CM files, using fallback method")
            return self._create_fallback_code_set()
        
        # Parse the files
        if not self.parse_icd10_files(year):
            logger.warning("Failed to parse ICD-10-CM files, using fallback method")
            return self._create_fallback_code_set()
        
        return self.get_all_codes()
    
    def _create_fallback_code_set(self) -> List[str]:
        """
        Create a fallback ICD-10-CM code set when download fails
        
        Returns:
            List of common ICD-10-CM codes
        """
        logger.info("Creating fallback ICD-10-CM code set...")
        
        # Generate common ICD-10-CM codes based on major categories
        fallback_codes = set()
        
        # Major categories
        categories = [
            ('A', 0, 99),    # Infectious and parasitic diseases
            ('B', 0, 99),    # Neoplasms
            ('C', 0, 99),    # Neoplasms
            ('D', 0, 99),    # Neoplasms
            ('E', 0, 99),    # Endocrine, nutritional and metabolic diseases
            ('F', 0, 99),    # Mental and behavioural disorders
            ('G', 0, 99),    # Diseases of the nervous system
            ('H', 0, 99),    # Diseases of the eye and adnexa
            ('I', 0, 99),    # Diseases of the circulatory system
            ('J', 0, 99),    # Diseases of the respiratory system
            ('K', 0, 99),    # Diseases of the digestive system
            ('L', 0, 99),    # Diseases of the skin and subcutaneous tissue
            ('M', 0, 99),    # Diseases of the musculoskeletal system
            ('N', 0, 99),    # Diseases of the genitourinary system
            ('O', 0, 99),    # Pregnancy, childbirth and the puerperium
            ('P', 0, 99),    # Certain conditions originating in the perinatal period
            ('Q', 0, 99),    # Congenital malformations, deformations and chromosomal abnormalities
            ('R', 0, 99),    # Symptoms, signs and abnormal clinical and laboratory findings
            ('S', 0, 99),    # Injury, poisoning and certain other consequences of external causes
            ('T', 0, 99),    # Injury, poisoning and certain other consequences of external causes
            ('U', 0, 99),    # Codes for special purposes
            ('V', 0, 99),    # External causes of morbidity
            ('W', 0, 99),    # External causes of morbidity
            ('X', 0, 99),    # External causes of morbidity
            ('Y', 0, 99),    # External causes of morbidity
            ('Z', 0, 99),    # Factors influencing health status and contact with health services
        ]
        
        # Generate codes for each category
        for letter, start, end in categories:
            for i in range(start, end + 1):
                # Add base code (e.g., A00)
                base_code = f"{letter}{i:02d}"
                fallback_codes.add(base_code)
                
                # Add decimal variants (e.g., A00.0, A00.00, A00.000, A00.0000)
                for decimal_places in range(1, 5):
                    decimal_code = f"{base_code}.{'0' * decimal_places}"
                    fallback_codes.add(decimal_code)
        
        # Add some common specific codes
        common_codes = [
            'I10', 'I10.1', 'I10.11', 'I10.111', 'I10.1111',
            'E11', 'E11.0', 'E11.00', 'E11.000', 'E11.0000',
            'C34', 'C34.0', 'C34.00', 'C34.000', 'C34.0000',
            'N18', 'N18.0', 'N18.00', 'N18.000', 'N18.0000',
            'F32', 'F32.0', 'F32.00', 'F32.000', 'F32.0000',
            'J44', 'J44.0', 'J44.00', 'J44.000', 'J44.0000',
            'M79', 'M79.0', 'M79.00', 'M79.000', 'M79.0000',
            'K59', 'K59.0', 'K59.00', 'K59.000', 'K59.0000',
            'L30', 'L30.0', 'L30.00', 'L30.000', 'L30.0000',
            'H52', 'H52.0', 'H52.00', 'H52.000', 'H52.0000',
        ]
        
        fallback_codes.update(common_codes)
        
        self.all_codes = fallback_codes
        logger.info(f"Created fallback set with {len(fallback_codes)} ICD-10-CM codes")
        
        return sorted(list(fallback_codes))


def get_icd10_codes(year: str = "2024", data_dir: str = "~/data/MIMIC_IV") -> List[str]:
    """
    Get comprehensive ICD-10-CM code set
    
    Args:
        year: Year of ICD-10-CM codes
        data_dir: Directory containing ICD-10-CM files
        
    Returns:
        List of all ICD-10-CM codes
    """
    manager = ICD10CodeManager(data_dir)
    return manager.create_comprehensive_code_set(year)


# Global cache for ICD-10 codes
_icd10_codes_cache = None

def get_cached_icd10_codes(year: str = "2024", data_dir: str = "~/data/MIMIC_IV") -> List[str]:
    """
    Get cached ICD-10-CM codes
    
    Args:
        year: Year of ICD-10-CM codes
        data_dir: Directory containing ICD-10-CM files
        
    Returns:
        List of ICD-10-CM codes
    """
    global _icd10_codes_cache
    if _icd10_codes_cache is None:
        _icd10_codes_cache = get_icd10_codes(year, data_dir)
    return _icd10_codes_cache

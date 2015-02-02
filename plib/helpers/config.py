import simplejson
import logging
from treedict import TreeDict

def configuration_file(filename, treedict=True):
    """
    Read a json file and returns its contents
    
    Args:
        filename (str):
            Full path to configuration file
            
    Keyw: 
        treedict (boolean):
            Return treedict object instead of dict (default:True)
    
    Returns (dict|treedict):
        Content of configuration files.
    """
    logging.info("File format configuration file: %s" % filename)
    with open(filename, 'r') as filehandle:
        data = simplejson.load(filehandle)

    if treedict:
        t = TreeDict.fromdict(data, expand_nested=True)
        return t
    else:
        return data
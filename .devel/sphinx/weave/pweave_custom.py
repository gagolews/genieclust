# NOTE: pweave is not in use anymore

# #!/usr/bin/env python3
#
# # ########################################################################## #
# #  Marek's output hooks hacks                                                #
# #  Copyleft (C) 2020-2025, Marek Gagolewski <https://www.gagolewski.com>     #
# # ########################################################################## #
#
#
# # NOTE! On stupid errors, make sure all modules are installed (e.g., tabulate!)
# # NOTE! Works with pweave==0.30
#
# import sys
# import pweave
#
#
# if len(sys.argv) != 3:
#     print("call: script infile outfile")
#     sys.exit()
#
#
# doc = pweave.Pweb(sys.argv[1], informat=None, doctype="sphinx",
#     kernel="python3", output=sys.argv[2], figdir="figures",
#     mimetype=None
# )
#
# doc.documentationmode = False
# pweave.rcParams["usematplotlib"] = True
# pweave.rcParams["cachedir"] = "cache"
# pweave.rcParams["storeresults"] = False
#
#
# class PwebSphinxFormatter2(pweave.PwebSphinxFormatter):
#     def initformat(self):
#         pweave.PwebSphinxFormatter.initformat(self)
#         self.formatdict["indent"] = '    ## '
#
#
# # doc.formatter = PwebSphinxFormatter2(True)
# # doc.formatter.formatdict["indent"] = '    ## '
#
#
# # ugly nasty filthy hacks :))
#
# def format_text_result2(self, text, chunk):
#     chunk["result"] = text
#     result = ""
#     if "%s" in chunk["outputstart"]:
#         chunk["outputstart"] = chunk["outputstart"] % self.language
#     if "%s" in chunk["termstart"]:
#         chunk["termstart"] = chunk["termstart"] % self.language
#
#     # Other things than term
#     if chunk['results'] == 'verbatim':
#         if len(chunk['result'].strip()) > 0:
#             if chunk["wrap"] is True or chunk['wrap'] == 'results' or chunk['wrap'] == 'output':
#                 chunk['result'] = self._wrap(chunk["result"])
#             chunk['result'] = "\n%s\n" % chunk["result"].rstrip()
#             chunk['result'] = self._indent2(chunk['result'])  # !!!!!!!!!
#             # chunk["result"] = self.fix_linefeeds(chunk['result'])
#             result += '%(outputstart)s%(result)s%(outputend)s' % chunk
#     elif chunk['results'] != 'verbatim':
#         result += self.fix_linefeeds(text)
#
#     return(result)
#
#
# def _indent2(self, text):
#     """Indent blocks for formats where indent is significant"""
#     if not text.startswith("\n"):
#         text = "\n" + text
#     if text.endswith("\n"):
#         text = text[:-1]
#     return text.replace('\n', '\n' + self.formatdict['indent'] + "## ")  # !!!!
#
#
# def formatfigure2(self, chunk):
#     fignames = chunk['figure']
#     caption = chunk['caption']
#     width = chunk['width']
#     result = ""
#     figstring = ""
#
#     for fig in fignames:
#         figstring += ('.. image:: %s\n   :width: %s\n\n' % (fig, width))
#
#     if chunk['figure'] and chunk['caption']:
#         result += (".. figure:: %s\n" \
#                     "   :width: %s\n\n" \
#                     "   %s\n\n" % (fignames[0], width, caption))
#     else:
#         result += figstring
#     return result
#
# doc.formatter.__class__.format_text_result = format_text_result2
# doc.formatter.__class__._indent2 = _indent2
# #doc.formatter.__class__.formatfigure = formatfigure2
#
# doc.weave()

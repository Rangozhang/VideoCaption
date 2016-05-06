__author__ = 'tylin'
from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider

class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        # print(imgIds)
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        # print(len(imgIds)) ## 676476 ids; 1000 in total
        # print(self.coco.imgToAnns) ## key-value pairs
        for imgId in imgIds:
            # print(imgId)
            gts[imgId] = self.coco.imgToAnns[imgId] ## length = 5
            # print(len(gts[imgId]))
            # print(gts[imgId])
            res[imgId] = self.cocoRes.imgToAnns[imgId]
            # print(len(res[imgId]))
            # print(res[imgId])

        # =================================================
        # Set up scorers
        # =================================================
        print '===== tokenization... gts'
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        print '===== tokenization... res'
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print '===== computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(gts, res)
            # print(scores)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    print "%s: %0.3f"%(m, sc)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                print "%s: %0.3f"%(method, score)
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
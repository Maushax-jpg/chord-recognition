import circularPitchSpace as cps
import matplotlib.pyplot as plt

def plotVector(ax, z):
    ax.plot([0,z.real], [0,z.imag], '-x', color="r", markersize=4)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
4
def plotCpsPrediction(fig,gs,chroma,chroma_index):
    r_F,r_FR,r_TR,r_DR = cps.transformChroma(chroma)

    pitch_class_index = cps.estimateKey(chroma[chroma_index,:],0.1)
    ax = fig.add_subplot(gs[2, 0])
    cps.plotCircleOfFifths(ax)
    plotVector(ax,r_F[chroma_index]*1j)

    ax = fig.add_subplot(gs[2,1])
    cps.plotCircleOfFifthsRelated(ax,pitch_class_index)    
    plotVector(ax,r_FR[chroma_index,pitch_class_index]*1j)
    
    ax = fig.add_subplot(gs[2,2])
    cps.plotCircleOfThirdsRelated(ax,pitch_class_index)
    plotVector(ax,r_TR[chroma_index,pitch_class_index]*1j)
    ax = fig.add_subplot(gs[2,3])
    cps.plotCircleOfDiatonicRelated(ax,pitch_class_index)
    plotVector(ax,r_DR[chroma_index,pitch_class_index]*1j)

#include "LiveDebugVariables.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/CodeGenPassBuilder.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundleIterator.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocCommon.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Printable.h"
#include <_types/_uint16_t.h>
#include <cstdio>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <system_error>
#include <utility>
#include <vector>

#include "gurobi_c++.h"
#include "gurobi_c.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

static RegisterRegAlloc PboUpdatedRegAlloc("pbo-updated", "PBO updated register allocator", createPBOUpdatedRegisterAllocator);

typedef unsigned long long idx_t;

namespace {

class PBORegAllocUpdated : public MachineFunctionPass {
  MachineFunction *MachineFunction;
  LiveIntervals *LiveIntervals;
  SlotIndexes *SlotIndexes;
  LiveRegMatrix *LiveRegMatrix;
  
  const TargetInstrInfo *TargetInstrInfo;
  const TargetRegisterInfo *TargetRegisterInfo;
  MachineRegisterInfo *MachineRegisterInfo;

  MachineBlockFrequencyInfo *MachineBlockFrequencyInfo;

  std::vector<Register> VirtRegs;
  idx_t VirtRegCount;

  std::map<SlotIndex, std::set<idx_t>> VIdsAtIdx;

  GRBQuadExpr ObjectiveExpr;
  GRBModel *Model;

  idx_t VarCount;

  std::vector<std::vector<MCPhysReg>> PhysRegSet;
  std::vector<std::map<SlotIndex, std::map<MCPhysReg, GRBVar>>> PhysVars;
  std::vector<std::map<SlotIndex, GRBVar>> SpillVars;

  std::map<Register, std::map<SlotIndex, MCPhysReg>> RegAssigns;
  std::map<Register, int> StackCache;

  std::map<MCPhysReg, MCPhysReg> DestToSrc;
  std::map<MCPhysReg, idx_t> DestToId;
  std::map<MCPhysReg, MCPhysReg> Alias;

  std::vector<std::pair<MCPhysReg, idx_t>> SrcToZero;
  std::map<MCPhysReg, bool> Dests;
  std::set<MCPhysReg> Srcs;

public:
  PBORegAllocUpdated(const RegClassFilterFunc F = allocateAllRegClasses);
  StringRef getPassName() const override { return "Updated PBO Register Allocator"; };
  
  void getAnalysisUsage(AnalysisUsage &AUsage) const override;
  bool runOnMachineFunction(llvm::MachineFunction &MFunction) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::NoPHIs
    );
  }
  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::IsSSA
    );
  }
  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::NoVRegs
    );
  }


  void genStoreRestrictions(void);
  void genClashRestrictions(void);
  void genBlockRestrictions(void);
  void genTermsRestrictions(void);

  void genObjectiveFunction(void);

  void readPBOSolution(void);
  void addBlockLiveIns(void);
  void assignRegisters(void);


  GRBVar genVar(void);
  void genInstrRestrictions(const MachineInstr &, idx_t, bool);

  void spillToStack(MachineBasicBlock &, MachineBasicBlock::instr_iterator &, Register, MCPhysReg);
  void loadViaStack(MachineBasicBlock &, MachineBasicBlock::instr_iterator &, Register, MCPhysReg);
  void moveRegToReg(MachineBasicBlock &, MachineBasicBlock::instr_iterator &, MCPhysReg, MCPhysReg);
  void substOperand(MachineBasicBlock::instr_iterator &, SlotIndex);

  void logMove(MCPhysReg, MCPhysReg, idx_t);
  void addMoveInstr(MachineBasicBlock &, MachineBasicBlock::instr_iterator &);
  void resetManager(void);

  int getStackSlot(Register);

  static char ID;
};

char PBORegAllocUpdated::ID = 0;

} // namespace 

INITIALIZE_PASS_BEGIN(PBORegAllocUpdated, "regallocpboupdated", "Updated PBO Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_END(PBORegAllocUpdated, "regallocpboupdated", "Updated PBO Allocator", false, false)

PBORegAllocUpdated::PBORegAllocUpdated(RegClassFilterFunc F) : MachineFunctionPass(ID) {}

void PBORegAllocUpdated::getAnalysisUsage(AnalysisUsage &AUsage) const {
  AUsage.setPreservesCFG();
  AUsage.addRequired<llvm::LiveIntervals>();
  AUsage.addRequired<llvm::SlotIndexes>();
  AUsage.addRequired<llvm::MachineBlockFrequencyInfo>();
  AUsage.addPreserved<llvm::MachineBlockFrequencyInfo>();
  AUsage.addRequired<llvm::LiveRegMatrix>();
  AUsage.addPreserved<llvm::LiveRegMatrix>();
  AUsage.addRequired<llvm::VirtRegMap>();
  AUsage.addPreserved<llvm::VirtRegMap>();
  MachineFunctionPass::getAnalysisUsage(AUsage);
}

GRBVar PBORegAllocUpdated::genVar(void) {
  return Model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "x" + std::to_string(++VarCount));
}

void PBORegAllocUpdated::genInstrRestrictions(const MachineInstr &Instr, idx_t VirtId, bool Spillable) {
  SlotIndex InstrIdx = SlotIndexes->getInstructionIndex(Instr).getBaseIndex();

  std::map<SlotIndex, std::set<idx_t>>::iterator IdSetIt = VIdsAtIdx.find(InstrIdx);
  if (IdSetIt == VIdsAtIdx.end()) {
    IdSetIt = VIdsAtIdx.insert({InstrIdx, std::set<idx_t>{VirtId}}).first;
  } else if (IdSetIt->second.find(VirtId) != IdSetIt->second.end()) {
    return;
  }

  IdSetIt->second.insert(VirtId);

  PhysVars[VirtId][InstrIdx] = std::map<MCPhysReg, GRBVar>();
  idx_t PhysRegSetSize = PhysRegSet[VirtId].size();
  GRBLinExpr Constraint{};

  for (idx_t PhysId = 0; PhysId < PhysRegSetSize; ++PhysId) {
    GRBVar Var = genVar();
    PhysVars[VirtId][InstrIdx][PhysRegSet[VirtId][PhysId]] = Var;
    ObjectiveExpr.addTerm(-((PhysRegSetSize - PhysId) / (float) PhysRegSetSize), Var);
    Constraint += + 1 * Var;
  }

  if (Spillable) {
    GRBVar Var = genVar();
    SpillVars[VirtId][InstrIdx] = Var;
    Constraint += + 1 * Var;
  }

  Model->addConstr(Constraint, GRB_EQUAL, 1, "Store Restriction");

  for (MachineOperand Operand : Instr.operands()) {
    if (Operand.isRegMask()) {
      for (MCPhysReg PhysReg : PhysRegSet[VirtId]) {
        if (Operand.clobbersPhysReg(PhysReg)) {
          Model->addConstr(PhysVars[VirtId][InstrIdx][PhysReg], GRB_EQUAL, 0, "Implicit Restriction");
        }
      }
    }
    
    if (Operand.isReg()) {
      for (MCPhysReg PhysReg : PhysRegSet[VirtId]) {
        if (TargetRegisterInfo->regsOverlap(Operand.getReg(), PhysReg)) {
          Model->addConstr(PhysVars[VirtId][InstrIdx][PhysReg], GRB_EQUAL, 0, "Implicit Restriction");
        }
      }
    }
  }
}

void PBORegAllocUpdated::genStoreRestrictions(void) {
  PhysRegSet = {};
  PhysVars = {};
  SpillVars = {};
  VarCount = 0;

  for (idx_t VirtId = 0; VirtId < VirtRegCount; ++VirtId) {
    Register VirtReg = VirtRegs[VirtId];

    const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);
    LiveInterval &LiveInterval = LiveIntervals->getInterval(VirtReg);

    ArrayRef<MCPhysReg> PrefOrder = RegisterClass.getRawAllocationOrder(*MachineFunction);

    PhysRegSet.push_back(std::vector<MCPhysReg>());

    for (ArrayRef<MCPhysReg>::iterator PhysRegIt = PrefOrder.begin(); PhysRegIt != PrefOrder.end(); ++PhysRegIt) {
      if (!MachineRegisterInfo->isAllocatable(*PhysRegIt)) {
        continue;
      }

      if (LiveRegMatrix->checkInterference(LiveInterval, *PhysRegIt) != llvm::LiveRegMatrix::IK_Free) {
       continue;
      }

      PhysRegSet[VirtId].push_back(*PhysRegIt);
    }

    PhysVars.push_back(std::map<SlotIndex, std::map<MCPhysReg, GRBVar>>());
    SpillVars.push_back(std::map<SlotIndex, GRBVar>());

    for (MachineRegisterInfo::def_instr_iterator DefIt = MachineRegisterInfo->def_instr_begin(VirtReg);
         DefIt != MachineRegisterInfo::def_instr_end(); ++DefIt) {
      genInstrRestrictions(*DefIt, VirtId, false);
    }

    for (MachineRegisterInfo::use_instr_iterator UseIt = MachineRegisterInfo->use_instr_begin(VirtReg);
         UseIt != MachineRegisterInfo::use_instr_end(); ++UseIt) {
      genInstrRestrictions(*UseIt, VirtId, false);
    }

    for (llvm::LiveInterval::iterator SegIt = LiveInterval.begin(); SegIt != LiveInterval.end(); ++SegIt) {
      for (SlotIndex InstrIdx = SegIt->start.getBaseIndex(); InstrIdx <= SegIt->end; InstrIdx = InstrIdx.getNextIndex()) {
        MachineInstr *Instr = SlotIndexes->getInstructionFromIndex(InstrIdx);

        if (Instr == nullptr) {
          continue;
        }

        genInstrRestrictions(*Instr, VirtId, true);
      }
    }
  }
}

void PBORegAllocUpdated::genClashRestrictions(void) {
  Alias = std::map<MCPhysReg, MCPhysReg>();

  for (std::map<SlotIndex, std::set<idx_t>>::iterator EntryIt = VIdsAtIdx.begin(); EntryIt != VIdsAtIdx.end(); ++EntryIt) {
    for (std::set<idx_t>::iterator IdIt1 = EntryIt->second.begin(); IdIt1 != EntryIt->second.end(); ++IdIt1) {
      for (std::set<idx_t>::iterator IdIt2 = std::next(IdIt1, 1); IdIt2 != EntryIt->second.end(); ++IdIt2) {
        for (MCPhysReg PhysReg1 : PhysRegSet[*IdIt1]) {
          for (MCPhysReg PhysReg2 : PhysRegSet[*IdIt2]) {
            if (!TargetRegisterInfo->regsOverlap(PhysReg1, PhysReg2)) {
              continue;
            }

            Model->addConstr(PhysVars[*IdIt1][EntryIt->first][PhysReg1] + PhysVars[*IdIt2][EntryIt->first][PhysReg2], GRB_LESS_EQUAL, 1, "Clash Restriction");
            
            if (PhysReg1 != PhysReg2) {
              Alias[PhysReg1] = PhysReg2;
              Alias[PhysReg2] = PhysReg1;
            }
          }
        }
      }
    }
  }
}

void PBORegAllocUpdated::genBlockRestrictions(void) {
  for (MachineBasicBlock &BasicBlock : *MachineFunction) {
    if (BasicBlock.empty()) {
      continue;
    }

    MachineInstrBundleIterator<MachineInstr> LastInstrIt = BasicBlock.getLastNonDebugInstr();
    if (LastInstrIt->isTerminator()) {
      while (LastInstrIt != BasicBlock.instr_begin() && std::prev(LastInstrIt)->isTerminator()) {
        LastInstrIt--;
      }
    }
    SlotIndex LastInstrIdx = SlotIndexes->getInstructionIndex(*LastInstrIt).getBaseIndex();

    for (MachineBasicBlock *SuccBlock : BasicBlock.successors()) {
      if (SuccBlock->empty()) {
        continue;
      }

      MachineInstrBundleIterator<MachineInstr> FirstInstrIt = SuccBlock->getFirstNonDebugInstr();
      SlotIndex FirstInstrIdx = SlotIndexes->getInstructionIndex(*FirstInstrIt).getBaseIndex();

      for (idx_t LastId : VIdsAtIdx[LastInstrIdx]) {
        for (idx_t FirstId : VIdsAtIdx[FirstInstrIdx]) {
          if (LastId != FirstId) {
            continue;
          }

          for (MCPhysReg LastPhysReg : PhysRegSet[LastId]) {
            for (MCPhysReg FirstPhysReg : PhysRegSet[FirstId]) {
              if (LastPhysReg != FirstPhysReg) {
                continue;
              }

              Model->addConstr(PhysVars[LastId][LastInstrIdx][LastPhysReg], GRB_EQUAL, PhysVars[FirstId][FirstInstrIdx][FirstPhysReg], "Block Restriction");
            }
          }
        }
      }
    }
  }
}

void PBORegAllocUpdated::genTermsRestrictions(void) {
  for (MachineBasicBlock &BasicBlock: *MachineFunction) {
    if (BasicBlock.empty()) {
      continue;
    }

    MachineInstrBundleIterator<MachineInstr> FirstTermIt = BasicBlock.getFirstTerminator();

    if (FirstTermIt == BasicBlock.end()) {
      continue;
    }

    MachineBasicBlock::instr_iterator TermIt = std::next(FirstTermIt.getInstrIterator());

    if (TermIt == BasicBlock.instr_end()) {
      continue;
    }

    SlotIndex FirstTermIdx = SlotIndexes->getInstructionIndex(*FirstTermIt).getBaseIndex();

    while (TermIt != BasicBlock.instr_end() && TermIt->isTerminator()) {
      SlotIndex InstrIdx = SlotIndexes->getInstructionIndex(*TermIt).getBaseIndex();
      TermIt++;

      if (VIdsAtIdx.find(InstrIdx) == VIdsAtIdx.end()) {
        continue;
      }

      for (idx_t VirtId : VIdsAtIdx[InstrIdx]) {
        if (VIdsAtIdx[FirstTermIdx].find(VirtId) == VIdsAtIdx[FirstTermIdx].end()) {
          continue;
        }

        for (MCPhysReg PhysReg : PhysRegSet[VirtId]) {
          Model->addConstr(PhysVars[VirtId][FirstTermIdx][PhysReg], GRB_EQUAL, PhysVars[VirtId][InstrIdx][PhysReg], "Term Restriction");
        }
      }
    }
  }
}

void PBORegAllocUpdated::genObjectiveFunction(void) {
  for (MachineBasicBlock &BasicBlock : *MachineFunction) {
    if (BasicBlock.empty()) {
      continue;
    }

    double ExpectedExecutions = 1000 * MachineBlockFrequencyInfo->getBlockFreqRelativeToEntryBlock(&BasicBlock);

    std::map<idx_t, std::map<MCPhysReg, GRBVar>> LastInstrMap = {};
    std::map<idx_t, GRBVar> LastInstrSpill = {};

    for (MachineBasicBlock::instr_iterator InstrIt = BasicBlock.instr_begin(); InstrIt != BasicBlock.instr_end(); ++InstrIt) {
      if (InstrIt->isDebugInstr()) {
        continue;
      }

      SlotIndex InstrIdx = SlotIndexes->getInstructionIndex(*InstrIt).getBaseIndex();

      std::map<idx_t, std::map<MCPhysReg, GRBVar>> CurInstrMap = {};
      std::map<idx_t, GRBVar> CurInstrSpill = {};

      for (idx_t VirtId : VIdsAtIdx[InstrIdx]) {
        CurInstrMap[VirtId] = PhysVars[VirtId][InstrIdx];
        
        if (LastInstrMap.find(VirtId) == LastInstrMap.end()) {
          continue;
        }

        for (idx_t I = 0; I < PhysRegSet[VirtId].size(); ++I) {
          for (idx_t J = 0; J < PhysRegSet[VirtId].size(); ++J) {
            if (I == J) {
              continue;
            }

            ObjectiveExpr.addTerm(ExpectedExecutions, LastInstrMap[VirtId][PhysRegSet[VirtId][I]], CurInstrMap[VirtId][PhysRegSet[VirtId][J]]);
          }
        }

        bool Spillable = SpillVars[VirtId].find(InstrIdx) != SpillVars[VirtId].end();
        bool WasSpillable = LastInstrSpill.find(VirtId) != LastInstrSpill.end();

        if (!Spillable && !WasSpillable) {
          continue;
        }

        if (Spillable) {
          CurInstrSpill[VirtId] = SpillVars[VirtId][InstrIdx];

          for (idx_t I = 0; I < PhysRegSet[VirtId].size(); ++I) {
            ObjectiveExpr.addTerm(ExpectedExecutions * 10, CurInstrSpill[VirtId], LastInstrMap[VirtId][PhysRegSet[VirtId][I]]);
          }
        }

        if (WasSpillable) {
          for (idx_t I = 0; I < PhysRegSet[VirtId].size(); ++I) {
            ObjectiveExpr.addTerm(ExpectedExecutions * 10, LastInstrSpill[VirtId], CurInstrMap[VirtId][PhysRegSet[VirtId][I]]);
          }
        }
      }

      LastInstrMap = CurInstrMap;
      LastInstrSpill = CurInstrSpill;
    }
  }
}

void PBORegAllocUpdated::readPBOSolution(void) {
  Model->write("/Users/pierreyan/logs/" + MachineFunction->getName().str() + ".lp");
  std::ifstream SolFile("/Users/pierreyan/logs/" + MachineFunction->getName().str() + ".sol");
  if (SolFile.good()) {
    Model->read("/Users/pierreyan/logs/" + MachineFunction->getName().str() + ".sol");
  }
  Model->optimize();
  if (Model->get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
    Model->computeIIS();
    Model->write("/Users/pierreyan/logs/" + MachineFunction->getName().str() + ".ilp");
  }
  Model->write("/Users/pierreyan/logs/" + MachineFunction->getName().str() + ".sol");

  RegAssigns = std::map<Register, std::map<SlotIndex, MCPhysReg>>();

  for (idx_t VirtId = 0; VirtId < VirtRegCount; ++VirtId) {
    Register VirtReg = VirtRegs[VirtId];
    LiveInterval &LiveInterval = LiveIntervals->getInterval(VirtReg);

    RegAssigns[VirtReg] = std::map<SlotIndex, MCPhysReg>();

    for (llvm::LiveInterval::iterator SegIt = LiveInterval.begin(); SegIt != LiveInterval.end(); ++SegIt) {
      for (SlotIndex InstrIdx = SegIt->start.getBaseIndex(); InstrIdx <= SegIt->end; InstrIdx = InstrIdx.getNextIndex()) {
        MachineInstr *Instr = SlotIndexes->getInstructionFromIndex(InstrIdx);

        if (Instr == nullptr) {
          continue;
        }

        RegAssigns[VirtReg][InstrIdx] = 0;

        for (MCPhysReg PhysReg : PhysRegSet[VirtId]) {
          if (PhysVars[VirtId][InstrIdx][PhysReg].get(GRB_DoubleAttr_X) == 1.0) {
            RegAssigns[VirtReg][InstrIdx] = PhysReg;
            break;
          }
        }
      }
    }
  }
}

void PBORegAllocUpdated::addBlockLiveIns(void) {
  for (MachineBasicBlock &BasicBlock : *MachineFunction) {
    if (BasicBlock.empty()) {
      continue;
    }

    MachineInstrBundleIterator<MachineInstr> FirstInstr = BasicBlock.getFirstNonDebugInstr();
    SlotIndex FirstIdx = SlotIndexes->getInstructionIndex(*FirstInstr).getBaseIndex();

    for (idx_t VirtId : VIdsAtIdx[FirstIdx]) {
      MCPhysReg PhysReg = RegAssigns[VirtRegs[VirtId]][FirstIdx];
      if (PhysReg == 0) {
        continue;
      }

      BasicBlock.addLiveIn(PhysReg);
    }
  }
}

void PBORegAllocUpdated::substOperand(MachineBasicBlock::instr_iterator &Instr, SlotIndex InstrIdx) {
  for (MachineOperand &Operand : (*Instr).operands()) {
    if (!Operand.isReg()) {
      continue;
    }

    Register Reg = Operand.getReg();

    if (Reg.isPhysical()) {
      continue;
    }

    Operand.substPhysReg(RegAssigns[Reg][InstrIdx], *TargetRegisterInfo);
  }
}

void PBORegAllocUpdated::logMove(MCPhysReg Src, MCPhysReg Dest, idx_t VirtId) {
  LLVM_DEBUG(dbgs() << "Move from " << printReg(Src, TargetRegisterInfo) << " to " << printReg(Dest, TargetRegisterInfo) << "\n");
  if (Dest == 0) {
    SrcToZero.push_back(std::pair<MCPhysReg, idx_t>{Src, VirtId});
    Srcs.insert(Src);

    if (Dests.find(Src) != Dests.end()) {
      Dests[Src] = true;
    } else if (Alias.find(Src) != Alias.end() && Dests.find(Alias[Src]) != Dests.end()) {
      Dests[Alias[Src]] = true;
    }
    return;
  }

  bool SrcOverlap = false;
  if (Dests.find(Src) != Dests.end()) {
    Dests[Src] = true;
    SrcOverlap = true;
  } else if (Alias.find(Src) != Alias.end() && Dests.find(Alias[Src]) != Dests.end()) {
    Dests[Alias[Src]] = true;
    SrcOverlap = true;
  }

  bool DestOverlap = false;
  if (Srcs.find(Dest) != Srcs.end() || (Alias.find(Dest) != Alias.end() && Srcs.find(Alias[Dest]) != Srcs.end())) {
    Dests[Dest] = true;
    DestOverlap = true;
  }

  LLVM_DEBUG(dbgs() << "Src: " << SrcOverlap << " Dest: " << DestOverlap << "\n");
  if (Alias.find(Src) != Alias.end()) {
    LLVM_DEBUG(dbgs() << "SrcAlias: " << printReg(Alias[Src], TargetRegisterInfo) << "\n");
  }
  if (Alias.find(Dest) != Alias.end()) {
    LLVM_DEBUG(dbgs() << "DestAlias: " << printReg(Alias[Dest], TargetRegisterInfo) << "\n");
  }

  if (!DestOverlap) {
    Dests[Dest] = false;
  }

  if (SrcOverlap && DestOverlap) {
    MCPhysReg Traveler = Src;
    MCPhysReg Lagger = 0;

    while (true) {
      if (DestToSrc.find(Traveler) != DestToSrc.end()) {
        Lagger = Traveler;
        Traveler = DestToSrc[Traveler];
      } else if (Alias.find(Traveler) != Alias.end() && DestToSrc.find(Alias[Traveler]) != DestToSrc.end()) {
        Lagger = Alias[Traveler];
        Traveler = DestToSrc[Alias[Traveler]];
      } else {
        break;
      }
    }

    if (Traveler == Dest) {
      Dests[Dest] = false;
    }
    if (Alias.find(Dest) != Alias.end() && Traveler == Alias[Dest]) {
      Dests[Dest] = false;
      DestToId[Traveler] = DestToId[Lagger];
    }
  }

  DestToSrc[Dest] = Src;
  DestToId[Dest] = VirtId;
  Srcs.insert(Src);
}

void PBORegAllocUpdated::spillToStack(MachineBasicBlock &BasicBlock, MachineBasicBlock::instr_iterator &InsertInstrIt, Register VirtReg, MCPhysReg PhysReg) {
  int SlotId = getStackSlot(VirtReg);
  const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

  LLVM_DEBUG(dbgs() << "Inserted spill from " << printReg(PhysReg, TargetRegisterInfo) << "\n");

  TargetInstrInfo->storeRegToStackSlot(BasicBlock, InsertInstrIt, PhysReg, true, SlotId, &RegisterClass, TargetRegisterInfo, Register());
  SlotIndexes->insertMachineInstrInMaps(*std::prev(InsertInstrIt));
}

void PBORegAllocUpdated::loadViaStack(MachineBasicBlock &BasicBlock, MachineBasicBlock::instr_iterator &InsertInstrIt, Register VirtReg, MCPhysReg PhysReg) {
  int SlotId = getStackSlot(VirtReg);
  const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

  LLVM_DEBUG(dbgs() << "Inserted load to " << printReg(PhysReg, TargetRegisterInfo) << "\n");

  TargetInstrInfo->loadRegFromStackSlot(BasicBlock, InsertInstrIt, PhysReg, SlotId, &RegisterClass, TargetRegisterInfo, Register());
  SlotIndexes->insertMachineInstrInMaps(*std::prev(InsertInstrIt));
}

void PBORegAllocUpdated::moveRegToReg(MachineBasicBlock &BasicBlock, MachineBasicBlock::instr_iterator &InsertInstrIt, MCPhysReg Src, MCPhysReg Dest) {
  LLVM_DEBUG(dbgs() << "Exe Move from " << printReg(Src, TargetRegisterInfo) << " to " << printReg(Dest, TargetRegisterInfo) << "\n");
  TargetInstrInfo->copyPhysReg(BasicBlock, InsertInstrIt, InsertInstrIt->getDebugLoc(), Dest, Src, true);
  SlotIndexes->insertMachineInstrInMaps(*std::prev(InsertInstrIt));
}

void PBORegAllocUpdated::addMoveInstr(MachineBasicBlock &BasicBlock, MachineBasicBlock::instr_iterator &InsertInstrIt) {
  LLVM_DEBUG(dbgs() << "Start Insert\n");
  for (std::pair<MCPhysReg, idx_t> SrcIdPair : SrcToZero) {
    LLVM_DEBUG(dbgs() << "To zero: " << printReg(SrcIdPair.first, TargetRegisterInfo) << "\n");
    spillToStack(BasicBlock, InsertInstrIt, VirtRegs[SrcIdPair.second], SrcIdPair.first);

    MCPhysReg Traveler = SrcIdPair.first, SrcReg;
    while (true) {
      if (DestToSrc.find(Traveler) == DestToSrc.end()) {
        if (Alias.find(Traveler) != Alias.end() && DestToSrc.find(Alias[Traveler]) != DestToSrc.end()) {
          Traveler = Alias[Traveler];
        } else {
          break;
        }
      }

      SrcReg = DestToSrc[Traveler];

      if (SrcReg == 0) {
        loadViaStack(BasicBlock, InsertInstrIt, VirtRegs[DestToId[Traveler]], Traveler);
      } else {
        moveRegToReg(BasicBlock, InsertInstrIt, DestToSrc[Traveler], Traveler);
      }

      Dests[Traveler] = true;
      Traveler = SrcReg;
    }
  }

  for (std::pair<MCPhysReg, bool> Start : Dests) {
    LLVM_DEBUG(dbgs() << "Starts: " << printReg(Start.first, TargetRegisterInfo) << "\n");
    if (Start.second) {
      continue;
    }

    std::vector<MCPhysReg> Stack = {};
    
    MCPhysReg Traveler = Start.first, SrcReg; bool Cycle = false;
    while (true) {
      Dests[Traveler] = true;
      Stack.push_back(Traveler);
      SrcReg = DestToSrc[Traveler];

      if (SrcReg == 0) {
        break;
      }
      if (SrcReg == Start.first) {
        Cycle = true;
        break;
      }

      Traveler = SrcReg;

      if (DestToSrc.find(Traveler) == DestToSrc.end()) {
        if (Alias.find(Traveler) != Alias.end() && DestToSrc.find(Alias[Traveler]) != DestToSrc.end()) {
          Stack.push_back(Traveler);
          Traveler = Alias[Traveler];

          if (Traveler == Start.first) {
            Cycle = true;
            break;
          }
        } else {
          break;
        }
      }
    }

    if (Cycle) {
      LLVM_DEBUG(dbgs() << "CYCLE\n");
      spillToStack(BasicBlock, InsertInstrIt, VirtRegs[DestToId[Start.first]], Start.first);

      for (idx_t Idx = 0; Idx < Stack.size() - 1; ++Idx) {
        if (Alias.find(Stack[Idx]) != Alias.end() && Alias[Stack[Idx]] == Stack[Idx+1]) {
          continue;
        }
        moveRegToReg(BasicBlock, InsertInstrIt, Stack[Idx+1], Stack[Idx]);
      }

      MCPhysReg LastReg = Stack[Stack.size()-1];
      loadViaStack(BasicBlock, InsertInstrIt, VirtRegs[DestToId[LastReg]], LastReg);
    } else {
      for (idx_t Idx = 0; Idx < Stack.size() - 1; ++Idx) {
        if (Alias.find(Stack[Idx]) != Alias.end() && Alias[Stack[Idx]] == Stack[Idx+1]) {
          continue;
        }
        moveRegToReg(BasicBlock, InsertInstrIt, Stack[Idx+1], Stack[Idx]);
      }

      MCPhysReg LastReg = Stack[Stack.size()-1];
      
      if (DestToSrc[LastReg] == 0) {
        loadViaStack(BasicBlock, InsertInstrIt, VirtRegs[DestToId[LastReg]], LastReg);
      } else {
        moveRegToReg(BasicBlock, InsertInstrIt, DestToSrc[LastReg], LastReg);
      }
    }
  }
}

void PBORegAllocUpdated::resetManager(void) {
  DestToSrc = {};
  DestToId = {};
  
  SrcToZero = {};
  Dests = {};
  Srcs = {};
}

int PBORegAllocUpdated::getStackSlot(Register VirtReg) {
  if (StackCache[VirtReg] != -1) {
    return StackCache[VirtReg];
  }

  const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

  idx_t Size = TargetRegisterInfo->getSpillSize(RegisterClass);
  Align Align = TargetRegisterInfo->getSpillAlign(RegisterClass);

  int FrameIdx = MachineFunction->getFrameInfo().CreateSpillStackObject(Size, Align);

  StackCache[VirtReg] = FrameIdx;
  return FrameIdx;
}

void PBORegAllocUpdated::assignRegisters(void) {
  for (MachineBasicBlock &BasicBlock : *MachineFunction) {
    std::map<idx_t, MCPhysReg> LastInstrMap = {};

    for (MachineBasicBlock::instr_iterator InstrIt = BasicBlock.instr_begin(); InstrIt != BasicBlock.instr_end(); ++InstrIt) {
      if (InstrIt->isDebugInstr()) {
        continue;
      }

      SlotIndex InstrIdx = SlotIndexes->getInstructionIndex(*InstrIt).getBaseIndex();

      substOperand(InstrIt, InstrIdx);

      LLVM_DEBUG(dbgs() << "Starting at "; InstrIt->print(dbgs()));

      MachineBasicBlock::instr_iterator InsertInstrIt = InstrIt;
      if (InsertInstrIt->isTerminator()) {
        while (InsertInstrIt != BasicBlock.instr_begin() && std::prev(InsertInstrIt)->isTerminator()) {
          InsertInstrIt--;
        }
      }

      resetManager();
      std::map<idx_t, MCPhysReg> CurInstrMap = {};

      for (idx_t VirtId : VIdsAtIdx[InstrIdx]) {
        CurInstrMap[VirtId] = RegAssigns[VirtRegs[VirtId]][InstrIdx];

        if (LastInstrMap.find(VirtId) == LastInstrMap.end()) {
          continue;
        }

        if (CurInstrMap[VirtId] == LastInstrMap[VirtId]) {
          continue;
        }

        logMove(LastInstrMap[VirtId], CurInstrMap[VirtId], VirtId);
      }

      LastInstrMap = CurInstrMap;

      addMoveInstr(BasicBlock, InsertInstrIt);
    }
  }
}

GRBEnv Env = GRBEnv();

bool PBORegAllocUpdated::runOnMachineFunction(llvm::MachineFunction &MFunction) {
  MachineFunction = &MFunction;
  LiveIntervals = &getAnalysis<llvm::LiveIntervals>();
  SlotIndexes = &getAnalysis<llvm::SlotIndexes>();
  LiveRegMatrix = &getAnalysis<llvm::LiveRegMatrix>();

  TargetInstrInfo = MachineFunction->getSubtarget().getInstrInfo();
  TargetRegisterInfo = MachineFunction->getSubtarget().getRegisterInfo();
  MachineRegisterInfo = &MachineFunction->getRegInfo();

  MachineBlockFrequencyInfo = &getAnalysis<llvm::MachineBlockFrequencyInfo>();

  VirtRegs = std::vector<Register>{};
  VirtRegCount = MachineRegisterInfo->getNumVirtRegs();
  VIdsAtIdx = std::map<SlotIndex, std::set<idx_t>>();

  StackCache = std::map<Register, int>();

  for (idx_t Idx = 0; Idx < VirtRegCount; ++Idx) {
    Register VirtReg = Register::index2VirtReg(Idx);

    VirtRegs.push_back(VirtReg);
    StackCache[VirtReg] = -1;
  }


  ObjectiveExpr = GRBQuadExpr();

  GRBModel Solver = GRBModel(Env);
  Model = &Solver;
  Model->getEnv().set(GRB_DoubleParam_TimeLimit, 1200);


  genStoreRestrictions();
  genClashRestrictions();
  genBlockRestrictions();
  genTermsRestrictions();


  genObjectiveFunction();
  Model->setObjective(ObjectiveExpr);


  readPBOSolution();
  addBlockLiveIns();
  assignRegisters();


  for (Register VirtReg : VirtRegs) {
    LiveIntervals->removeInterval(VirtReg);
  }

  MachineRegisterInfo->clearVirtRegs();

  return true;
}

FunctionPass *llvm::createPBOUpdatedRegisterAllocator() {
  return new PBORegAllocUpdated();
}

FunctionPass *llvm::createPBOUpdatedRegisterAllocator(RegClassFilterFunc F) {
  return new PBORegAllocUpdated(F);
}
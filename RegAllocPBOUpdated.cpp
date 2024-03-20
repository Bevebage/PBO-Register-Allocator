#include "LiveDebugVariables.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
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
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocCommon.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <_types/_uint16_t.h>
#include <cstdio>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "regalloc"
#define SOLVER_COMMAND "gurobi_cl ResultFile=solution.sol"

static RegisterRegAlloc PboUpdatedRegAlloc("pbo-updated", "PBO updated register allocator", createPBOUpdatedRegisterAllocator);

typedef unsigned long long idx_t;
typedef std::map<SlotIndex, std::set<idx_t>> slot_queue_t;

namespace {

class PBORegAllocUpdated : public MachineFunctionPass {

  MachineFunction *MachineFunction;
  LiveIntervals *LiveIntervals;
  LiveRegMatrix *LiveRegMatrix;
  MachineRegisterInfo *MachineRegisterInfo;
  SlotIndexes *SlotIndexes;
  MachineBlockFrequencyInfo *MachineBlockFrequencyInfo;
  const TargetRegisterInfo *TargetRegisterInfo;
  const TargetInstrInfo *TargetInstrInfo;

  std::vector<Register> VirtRegs;
  idx_t VirtRegCount;

  slot_queue_t SlotQueue;

  std::ofstream OpbFile;
  std::fstream ConstraintFile;
  std::string ObjectiveFunction;

  idx_t LogVarCount;
  idx_t ConstraintCount;

  std::map<Register, idx_t> VirtIdxes;

  std::vector<std::vector<MCPhysReg>> PhysSets;
  std::vector<std::map<SlotIndex, std::map<MCPhysReg, std::string>>> PhysLogVarSets;
  std::vector<std::map<SlotIndex, std::string>> SpillLogVarSets;

  std::map<std::string, bool> LogVarAssignments;
  std::vector<std::map<SlotIndex, MCPhysReg>> RegAssignments;
  std::map<Register, int> StackMapping;

  // Instruction Insertion Stuff
  std::map<MCPhysReg, MCPhysReg> DestToSrc;
  std::map<MCPhysReg, idx_t> DestToReg;

  std::vector<std::pair<MCPhysReg, idx_t>> ToZero;
  std::set<MCPhysReg> Starts;

public:
  PBORegAllocUpdated(const RegClassFilterFunc F = allocateAllRegClasses);

  StringRef getPassName() const override { return "Updated PBO Register Allocator"; };

  void getAnalysisUsage(AnalysisUsage &AUsage) const override;

  bool runOnMachineFunction(class MachineFunction &MF) override;

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

  std::string genLogVar(void);
  void instrConditions(const MachineInstr &, idx_t, bool);

  void genStoreConditions(void);
  void genConflictConditions(void);
  void genEndingRestrictions(void);

  void genObjectiveFunction(void);
  void mixSwapPenalty(void);

  void genAssignments();
  void substOperand(MachineBasicBlock::instr_iterator&, SlotIndex);
  void assignPhysRegisters(void);

  int getStackSlot(Register Reg);

  static char ID;

  // Instruction Insertion Stuff
  void addEdge(std::pair<MCPhysReg, MCPhysReg>, idx_t);
  void insertInstrs(MachineBasicBlock &, MachineBasicBlock::instr_iterator &);
  void clearQueue(void);
};

char PBORegAllocUpdated::ID = 0;

} // namespace

INITIALIZE_PASS_BEGIN(PBORegAllocUpdated, "regallocpboupdated", 
                      "Updated PBO Allocator", false, false)
INITIALIZE_PASS_END(PBORegAllocUpdated, "regallocpboupdated", 
                    "Updated PBO Allocator", false, false)

PBORegAllocUpdated::PBORegAllocUpdated(RegClassFilterFunc F) : MachineFunctionPass(ID) {}

void PBORegAllocUpdated::getAnalysisUsage(AnalysisUsage &AUsage) const {
  AUsage.setPreservesCFG();
  AUsage.addRequired<llvm::LiveIntervals>();
  AUsage.addPreserved<llvm::LiveIntervals>();
  AUsage.addPreserved<llvm::SlotIndexes>();
  AUsage.addRequired<llvm::SlotIndexes>();
  AUsage.addRequired<llvm::MachineBlockFrequencyInfo>();
  AUsage.addPreserved<llvm::MachineBlockFrequencyInfo>();
  AUsage.addRequired<llvm::LiveRegMatrix>();
  AUsage.addPreserved<llvm::LiveRegMatrix>();
  MachineFunctionPass::getAnalysisUsage(AUsage);
}

std::string PBORegAllocUpdated::genLogVar(void) {
  return "x" + std::to_string(++LogVarCount);
}

void PBORegAllocUpdated::instrConditions(const MachineInstr &Instr, idx_t VirtIdx, bool Spillable) {
  SlotIndex SegIdx = SlotIndexes->getInstructionIndex(Instr);

  slot_queue_t::iterator RegGroup = SlotQueue.find(SegIdx);

  if (RegGroup == SlotQueue.end()) {
    RegGroup = SlotQueue.insert({SegIdx, std::set<idx_t>({VirtIdx})}).first;
  } else if (RegGroup->second.find(VirtIdx) != RegGroup->second.end()) {
    return;
  }

  RegGroup->second.insert(VirtIdx);

  PhysLogVarSets[VirtIdx][SegIdx] = std::map<MCPhysReg, std::string>();

  idx_t PhysRegCount = PhysSets[VirtIdx].size();

  for (size_t PhysIdx = 0; PhysIdx < PhysRegCount; ++PhysIdx) {
    std::string LogVar = genLogVar();
    PhysLogVarSets[VirtIdx][SegIdx][PhysSets[VirtIdx][PhysIdx]] = LogVar;
    if (PhysIdx < 3) {
      ObjectiveFunction += " -1 " + LogVar;
    }
    ConstraintFile << " +1 " << LogVar;
  }

  if (Spillable) {
    std::string LogVar = genLogVar();
    SpillLogVarSets[VirtIdx][SegIdx] = LogVar;
    ConstraintFile << " +1 " << LogVar;
  }

  ConstraintFile << " = 1;\n";
  ConstraintCount++;

  // Implicit Def Conditions

  const MCInstrDesc &Desc = TargetInstrInfo->get(Instr.getOpcode());

  for (MCPhysReg PhysReg : Desc.implicit_defs()) {
    if (PhysLogVarSets[VirtIdx][SegIdx].find(PhysReg) == PhysLogVarSets[VirtIdx][SegIdx].end()) {
      continue;
    }

    ConstraintFile << " +1 " << PhysLogVarSets[VirtIdx][SegIdx][PhysReg] << " = 0;\n";
    ConstraintCount++;
  }
}

void PBORegAllocUpdated::genStoreConditions(void) {
  idx_t VirtIdx = 0;
  for (Register VirtReg : VirtRegs) {
    VirtIdxes[VirtReg] = VirtIdx;

    const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);
    ArrayRef<uint16_t> RawOrder = RegisterClass.getRawAllocationOrder(*MachineFunction);

    LiveInterval &LiveInterval = LiveIntervals->getInterval(VirtReg);

    PhysSets.push_back(std::vector<MCPhysReg>());

    for (ArrayRef<uint16_t>::iterator PhysRegIt = RawOrder.begin(); PhysRegIt != RawOrder.end(); ++PhysRegIt) {
      if (MachineRegisterInfo->isReserved(*PhysRegIt)) {
        continue;
      }
      if (MachineRegisterInfo->isAllocatable(*PhysRegIt)) {
        PhysSets[VirtIdx].push_back(*PhysRegIt);
      } 
    }

    PhysLogVarSets.push_back(std::map<SlotIndex, std::map<MCPhysReg, std::string>>());
    SpillLogVarSets.push_back(std::map<SlotIndex, std::string>());

    for (MachineRegisterInfo::def_instr_iterator DefIt = MachineRegisterInfo->def_instr_begin(VirtReg);
         DefIt != MachineRegisterInfo::def_instr_end(); ++DefIt) {
      instrConditions(*DefIt, VirtIdx, false);
    }

    for (MachineRegisterInfo::use_instr_iterator UseIt = MachineRegisterInfo->use_instr_begin(VirtReg);
         UseIt != MachineRegisterInfo::use_instr_end(); ++UseIt) {
      instrConditions(*UseIt, VirtIdx, false);
    }

    for (llvm::LiveInterval::iterator IntIt = LiveInterval.begin(); IntIt != LiveInterval.end(); ++IntIt) {
      for (SlotIndex SegIdx = IntIt->start.getBaseIndex(); SegIdx <= IntIt->end; SegIdx = SegIdx.getNextIndex()) {
        MachineInstr *Instr = SlotIndexes->getInstructionFromIndex(SegIdx);

        if (Instr == nullptr) {
          continue;
        }

        instrConditions(*Instr, VirtIdx, true);        
      }
    }

    VirtIdx++;
  }
}

void PBORegAllocUpdated::genConflictConditions(void) {
  ConstraintFile << "* Confliction Restrictions\n";
  for (slot_queue_t::iterator SlotIt = SlotQueue.begin(); SlotIt != SlotQueue.end(); ++SlotIt) {
    for (std::set<idx_t>::iterator RegIt1 = SlotIt->second.begin(); RegIt1 != SlotIt->second.end(); ++RegIt1) {
      for (std::set<idx_t>::iterator RegIt2 = std::next(RegIt1, 1); RegIt2 != SlotIt->second.end(); ++RegIt2) {
        for (MCPhysReg PhysReg1 : PhysSets[*RegIt1]) {
          for (MCPhysReg PhysReg2 : PhysSets[*RegIt2]) {
            if (PhysReg1 != PhysReg2) {
              continue;
            }

            ConstraintFile << " +1 " << PhysLogVarSets[*RegIt1][SlotIt->first][PhysReg1] 
                           << " +1 " << PhysLogVarSets[*RegIt2][SlotIt->first][PhysReg2]
                           << " <= 1;\n";

            ConstraintCount++;
          }
        }
      }
    }
  }
}

void PBORegAllocUpdated::genEndingRestrictions(void) {
  ConstraintFile << "* Ending Restrictions\n";
  for (MachineBasicBlock &BasicBlock : *MachineFunction) {
    SlotIndex LastIdx = SlotIndexes->getInstructionIndex(BasicBlock.instr_back()).getBaseIndex();
    std::set<idx_t> *ExitLiveVirts = &SlotQueue[LastIdx];

    for (MachineBasicBlock *Successor : BasicBlock.successors()) {

      SlotIndex FirstIdx = SlotIndexes->getInstructionIndex(Successor->instr_front()).getBaseIndex();
      std::set<idx_t> *EntryLiveVirts = &SlotQueue[FirstIdx];

      for (idx_t ExitIdx : *ExitLiveVirts) {
        for (idx_t EntryIdx : *EntryLiveVirts) {
          if (ExitIdx != EntryIdx) {
            continue;
          }


          for (MCPhysReg PhysReg1 : PhysSets[ExitIdx]) {
            for (MCPhysReg PhysReg2 : PhysSets[EntryIdx]) {
              if (PhysReg1 != PhysReg2) {
                continue;
              }

              ConstraintFile << " +1 " << PhysLogVarSets[ExitIdx][LastIdx][PhysReg1]
                             << " -1 " << PhysLogVarSets[EntryIdx][FirstIdx][PhysReg2]
                             << " = 0;\n";
              ConstraintCount++;
            }
          }
        }
      }
    }
  }
}

void PBORegAllocUpdated::genObjectiveFunction(void) {
  idx_t VirtIdx = 0;
  for (Register VirtReg : VirtRegs) {
    for (MachineRegisterInfo::use_instr_iterator UseIt = MachineRegisterInfo->use_instr_begin(VirtReg);
         UseIt != MachineRegisterInfo::use_instr_end(); ++UseIt) {
      int Expected = MachineBlockFrequencyInfo->getBlockFreq(UseIt->getParent()).getFrequency() / MachineBlockFrequencyInfo->getEntryFreq().getFrequency();
      SlotIndex UseIdx = SlotIndexes->getInstructionIndex(*UseIt);
      SlotIndex PrevIdx = UseIdx.getPrevIndex();

      for (MCPhysReg PhysReg : PhysSets[VirtIdx]) {
        ObjectiveFunction += " -" + std::to_string((int)Expected) + " " 
                          + PhysLogVarSets[VirtIdx][UseIdx][PhysReg] + " " 
                          + PhysLogVarSets[VirtIdx][PrevIdx][PhysReg];
      }
    }

    VirtIdx++;
  }
}

void PBORegAllocUpdated::mixSwapPenalty(void) {
  idx_t VirtIdx = 0;
  for (Register VirtReg : VirtRegs) {
    LiveInterval &LiveInterval = LiveIntervals->getInterval(VirtReg);

    for (llvm::LiveInterval::iterator IntIt = LiveInterval.begin(); IntIt != LiveInterval.end(); ++IntIt) {
      std::map<MCPhysReg, std::string> PrevMap = PhysLogVarSets[VirtIdx][IntIt->start.getBaseIndex()];
      for (SlotIndex SegIdx = IntIt->start.getNextIndex().getBaseIndex(); SegIdx <= IntIt->end; SegIdx = SegIdx.getNextIndex()) {
        MachineInstr *Instr = SlotIndexes->getInstructionFromIndex(SegIdx);

        if (Instr == nullptr) {
          continue;
        }

        int Expected = MachineBlockFrequencyInfo->getBlockFreq(Instr->getParent()).getFrequency() / MachineBlockFrequencyInfo->getEntryFreq().getFrequency();

        for (std::pair<MCPhysReg, std::string> Entry : PhysLogVarSets[VirtIdx][SegIdx]) {
          ObjectiveFunction += " -" + std::to_string((int)Expected * 50) + " " + Entry.second + " " + PrevMap[Entry.first];
        }

        PrevMap = PhysLogVarSets[VirtIdx][SegIdx];
      }
    }

    VirtIdx++;
  }
}

void PBORegAllocUpdated::genAssignments() {
  std::ifstream File("solution.sol");

  if (File.is_open()) {
    std::string Line;

    getline(File, Line);
    for (idx_t Reads = 0; Reads < LogVarCount && getline(File, Line); ++Reads) {
      std::stringstream Stream(Line);

      std::string Var, Assignment;
      Stream >> Var >> Assignment;

      LogVarAssignments[Var] = std::stoi(Assignment) != 0;
    }

    File.close();
  } else {
    LLVM_DEBUG(dbgs() << "Couldn't open solutions\n");
  }

  idx_t VirtIdx = 0;
  for (Register VirtReg : VirtRegs) {
    LiveInterval &LiveInterval = LiveIntervals->getInterval(VirtReg);

    RegAssignments.push_back(std::map<SlotIndex, MCPhysReg>());

    for (llvm::LiveInterval::iterator IntIt = LiveInterval.begin(); IntIt != LiveInterval.end(); ++IntIt) {
      for (SlotIndex SegIdx = IntIt->start.getBaseIndex(); SegIdx <= IntIt->end; SegIdx = SegIdx.getNextIndex()) {
        RegAssignments[VirtIdx][SegIdx] = 0;
        for (MCPhysReg PhysReg : PhysSets[VirtIdx]) {
          if (LogVarAssignments[PhysLogVarSets[VirtIdx][SegIdx][PhysReg]]) {
            RegAssignments[VirtIdx][SegIdx] = PhysReg;
            break;
          }
        }
      }
    }

    VirtIdx++;
  }
}

void PBORegAllocUpdated::substOperand(MachineBasicBlock::instr_iterator &Inst, SlotIndex InstIdx) {
  for (MachineOperand &Operand : (*Inst).operands()) {
    if (!Operand.isReg()) {
      continue;
    }

    Register Reg = Operand.getReg();

    if (!Register::isVirtualRegister(Reg)) {
      continue;
    }

    idx_t VirtIdx = VirtIdxes[Reg];

    LLVM_DEBUG(dbgs() << printReg(Reg) << " " << VirtIdx << " " << InstIdx << " " << printReg(RegAssignments[VirtIdx][InstIdx]) << "\n");

    Operand.substPhysReg(RegAssignments[VirtIdx][InstIdx], *TargetRegisterInfo);
  }
}

void PBORegAllocUpdated::addEdge(std::pair<MCPhysReg, MCPhysReg> Move, idx_t VirtIdx) {
  if (Move.second == 0) {
    ToZero.push_back(std::pair<MCPhysReg, idx_t>{Move.first, VirtIdx});
    return;
  }

  Starts.insert(Move.second);
  DestToSrc[Move.second] = Move.first;
  DestToReg[Move.second] = VirtIdx;
}

void PBORegAllocUpdated::insertInstrs(MachineBasicBlock &MachineBasicBlock, llvm::MachineBasicBlock::instr_iterator &InsertIt) {
  for (std::pair<MCPhysReg, idx_t> Blunt : ToZero) {
    Register VirtReg = VirtRegs[DestToReg[Blunt.second]];
    int FrameIdx = getStackSlot(VirtReg);
    const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

    TargetInstrInfo->storeRegToStackSlot(MachineBasicBlock, InsertIt, Blunt.first, true, FrameIdx, &RegisterClass, TargetRegisterInfo, Register());

    MCPhysReg CurReg = DestToSrc[Blunt.first];
    while (DestToSrc.find(CurReg) != DestToSrc.end()) {
      VirtReg = VirtRegs[DestToReg[Blunt.second]];
      FrameIdx = getStackSlot(VirtReg);

      MCPhysReg SrcReg = DestToSrc[CurReg];
      
      if (SrcReg != 0) {
        TargetInstrInfo->copyPhysReg(MachineBasicBlock, InsertIt, InsertIt->getDebugLoc(), CurReg,SrcReg, true);
      } else {
        const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);
        TargetInstrInfo->loadRegFromStackSlot(MachineBasicBlock, InsertIt, CurReg, FrameIdx, &RegisterClass, TargetRegisterInfo, Register());
      }

      Starts.erase(CurReg);
      CurReg = SrcReg;
    }
  }

  for (MCPhysReg Start : Starts) {
    std::vector<MCPhysReg> Stack = {};

    MCPhysReg CurReg = Start; bool Cycle = false;
    while (DestToSrc.find(CurReg) != DestToSrc.end()) {
      Stack.push_back(CurReg);
      MCPhysReg SrcReg = DestToReg[CurReg];

      if (SrcReg == 0) {
        break;
      }

      if (SrcReg == Start) {
        Cycle = true;
        break;
      }

      CurReg = SrcReg;
    }

    if (Cycle) {
      Register VirtReg = VirtRegs[DestToReg[Start]];
      int FrameIdx = getStackSlot(-1);
      const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

      TargetInstrInfo->storeRegToStackSlot(MachineBasicBlock, InsertIt, Start, true, FrameIdx, &RegisterClass, TargetRegisterInfo, Register());

      for (idx_t Idx = 0; Idx < Stack.size() - 1; ++Idx) {
        TargetInstrInfo->copyPhysReg(MachineBasicBlock, InsertIt, InsertIt->getDebugLoc(), Stack[Idx], Stack[Idx+1], true);
      }

      MCPhysReg LastReg = Stack[Stack.size()-1];

      VirtReg = VirtRegs[DestToReg[LastReg]];
      FrameIdx = getStackSlot(-1);
      const TargetRegisterClass &LastRegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

      TargetInstrInfo->loadRegFromStackSlot(MachineBasicBlock, InsertIt, LastReg, FrameIdx, &LastRegisterClass, TargetRegisterInfo, Register());
    } else {
      for (idx_t Idx = 0; Idx < Stack.size() - 1; ++Idx) {
        TargetInstrInfo->copyPhysReg(MachineBasicBlock, InsertIt, InsertIt->getDebugLoc(), Stack[Idx], Stack[Idx+1], true);
      }
      
      MCPhysReg LastReg = Stack[Stack.size()-1];
      if (DestToSrc[LastReg] == 0) {
        Register VirtReg = VirtRegs[DestToReg[LastReg]];
        int FrameIdx = getStackSlot(VirtReg);
        const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

        TargetInstrInfo->loadRegFromStackSlot(MachineBasicBlock, InsertIt, LastReg, FrameIdx, &RegisterClass, TargetRegisterInfo, Register());
      } else {
        TargetInstrInfo->copyPhysReg(MachineBasicBlock, InsertIt, InsertIt->getDebugLoc(), LastReg, DestToSrc[LastReg], true);
      }
    }
  }
}

void PBORegAllocUpdated::clearQueue(void) {
  DestToSrc = {};
  DestToReg = {};

  ToZero = {};
  Starts = {};
}

void PBORegAllocUpdated::assignPhysRegisters(void) {
  for (llvm::MachineBasicBlock &MachineBasicBlock : (*MachineFunction)) {
    std::map<idx_t, MCPhysReg> LiveMap = {};

    for (llvm::MachineBasicBlock::instr_iterator InstIt = MachineBasicBlock.instr_begin();
         InstIt != MachineBasicBlock.end(); ++InstIt) {
      
      SlotIndex InstIdx = SlotIndexes->getInstructionIndex(*InstIt).getBaseIndex();

      substOperand(InstIt, InstIdx);

      idx_t InstInserted = 0;

      // This is to ensure that we do not break up the terminator block at the end of a basic block
      MachineBasicBlock::instr_iterator InsertIt = InstIt;
      while (InsertIt->isTerminator() && InsertIt != MachineBasicBlock.instr_begin()) {
        InsertIt--;
      }

      clearQueue();

      // Moving values to and from stack slots or between registers, as needed
      for (idx_t VirtIdx : SlotQueue[InstIdx]) {
        // If new
        if (LiveMap.find(VirtIdx) == LiveMap.end()) {
          LiveMap[VirtIdx] = RegAssignments[VirtIdx][InstIdx];
          continue;
        }

        // If nothing changes
        if (RegAssignments[VirtIdx][InstIdx] == LiveMap[VirtIdx]) {
          continue;
        }
        
        addEdge(std::pair<MCPhysReg, MCPhysReg>{RegAssignments[VirtIdx][InstIdx], LiveMap[VirtIdx]}, VirtIdx);

        // Register VirtReg = VirtRegs[VirtIdx];

        // int FrameIdx = getStackSlot(VirtReg);
        // const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(VirtReg);

        // if (LiveMap[VirtIdx] == 0) {
        //   // VirtIdx was stored in the stack, need to load it into a physical register
        //   TargetInstrInfo->loadRegFromStackSlot(MachineBasicBlock, InsertIt, RegAssignments[VirtIdx][InstIdx], FrameIdx, &RegisterClass, TargetRegisterInfo, Register());
        // } 
        // else if (RegAssignments[VirtIdx][InstIdx] == 0) {
        //   // VirtIdx was in physical register, needs to be spilled to stack
        //   TargetInstrInfo->storeRegToStackSlot(MachineBasicBlock, InsertIt, LiveMap[VirtIdx], true, FrameIdx, &RegisterClass, TargetRegisterInfo, Register());
        // } 
        // else {
        //   // Need to move from one register into another
        //   TargetInstrInfo->copyPhysReg(MachineBasicBlock, InsertIt, InsertIt->getDebugLoc(), RegAssignments[VirtIdx][InstIdx], LiveMap[VirtIdx], true);
        // }

        InstInserted++;
        LiveMap[VirtIdx] = RegAssignments[VirtIdx][InstIdx];
      }

      insertInstrs(MachineBasicBlock, InsertIt);

      MachineBasicBlock::instr_iterator TmpIt = InsertIt;
      for (idx_t InstDelta = 1; InstDelta <= InstInserted; ++InstDelta) {
        TmpIt--;
        substOperand(TmpIt, InstIdx);
      }
    }
  }
}

int PBORegAllocUpdated::getStackSlot(Register Reg) {
  if (StackMapping[Reg] != -1) {
    return StackMapping[Reg];
  }

  const TargetRegisterClass &RegisterClass = *MachineRegisterInfo->getRegClass(Reg);

  idx_t Size = TargetRegisterInfo->getSpillSize(RegisterClass);
  Align Alignment = TargetRegisterInfo->getSpillAlign(RegisterClass);

  int FrameIdx = MachineFunction->getFrameInfo().CreateSpillStackObject(Size, Alignment);

  StackMapping[Reg] = FrameIdx;
  return FrameIdx;
}

bool PBORegAllocUpdated::runOnMachineFunction(class MachineFunction &MF) {
  MachineFunction = &MF;
  LiveIntervals = &getAnalysis<llvm::LiveIntervals>();
  LiveRegMatrix = &getAnalysis<llvm::LiveRegMatrix>();
  TargetRegisterInfo = MachineFunction->getSubtarget().getRegisterInfo();
  TargetInstrInfo = MachineFunction->getSubtarget().getInstrInfo();
  MachineRegisterInfo = &MachineFunction->getRegInfo();
  SlotIndexes = &getAnalysis<llvm::SlotIndexes>();
  MachineBlockFrequencyInfo = &getAnalysis<llvm::MachineBlockFrequencyInfo>();

  VirtRegs = std::vector<Register>();
  VirtRegCount = MachineRegisterInfo->getNumVirtRegs();

  StackMapping = std::map<Register, int>();

  for (idx_t Idx = 0; Idx < VirtRegCount; ++Idx) {
    Register VirtReg = Register::index2VirtReg(Idx);

    VirtRegs.push_back(VirtReg);
    StackMapping[VirtReg] = -1;
  }

  LogVarCount = 0;
  ConstraintCount = 0;

  ObjectiveFunction = "min:";

  std::string OpbFileName = MachineFunction->getName().str() + "_problem.opb";
  std::string ConFileName = MachineFunction->getName().str() + "_constraints.txt";

  OpbFile.open(OpbFileName, std::ofstream::out | std::ofstream::trunc);
  ConstraintFile.open(ConFileName, std::ofstream::out | std::ofstream::trunc);

  SlotQueue = slot_queue_t();
  VirtIdxes = std::map<Register, idx_t>();

  PhysSets = std::vector<std::vector<MCPhysReg>>();
  PhysLogVarSets = std::vector<std::map<SlotIndex, std::map<MCPhysReg, std::string>>>();
  SpillLogVarSets = std::vector<std::map<SlotIndex, std::string>>();

  genStoreConditions();
  genConflictConditions();
  genEndingRestrictions();

  // genObjectiveFunction();
  mixSwapPenalty();

  OpbFile << "*#variable= " << LogVarCount
          << " #constraint= " << ConstraintCount
          << "\n* constraints for function " << MachineFunction->getName().str()
          << "\n";
  OpbFile << ObjectiveFunction << ";\n";

  ConstraintFile.close();
  std::ifstream ConstraintInFile;
  ConstraintInFile.open(ConFileName);
  OpbFile << ConstraintInFile.rdbuf();
  ConstraintInFile.close();
  OpbFile.close();

  std::string SolverCmd(SOLVER_COMMAND);
  std::string Command;

  if (false) { // Timeout goes here
    // Command = SolverCmd = " Default " + std::to_string(Timeout) + " " + OpbFileName
  } else {
    Command = SolverCmd + " " + OpbFileName;
  }

  system(Command.c_str());

  LogVarAssignments = std::map<std::string, bool>();
  RegAssignments = std::vector<std::map<SlotIndex, MCPhysReg>>();

  genAssignments();
  assignPhysRegisters();

  return true;
}

FunctionPass *llvm::createPBOUpdatedRegisterAllocator() {
  return new PBORegAllocUpdated();
}

FunctionPass *llvm::createPBOUpdatedRegisterAllocator(RegClassFilterFunc F) {
  return new PBORegAllocUpdated(F);
}
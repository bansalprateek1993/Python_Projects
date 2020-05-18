//package jena.examples.rdf ;

import org.apache.jena.rdf.model.*;
import org.apache.jena.vocabulary.*;


/** Tutorial 3 Statement attribute accessor methods
 */
public class TitanicJena extends Object {
    public static void main (String args[]) {
    
        // some definitions
        String personSex    = "http://titanic/sex";
        String SexP = "sex";

        String personSurvival    = "http://titanic/survival";
        String SurviveP = "survival";

        String personClass    = "http://titanic/TicketClass";
        String ClassP = "TicketClass";

        String personAge    = "http://titanic/AgeInYears";
        String AgeP = "Age";

        String personSib    = "http://titanic/NumberOfSiblings";
        String SiblingsP = "Number of Siblings";

        String personPar    = "http://titanic/NumberOfParents";
        String ParentsP = "Number of Parents";

        String TicketNumber    = "http://titanic/TicketNumber";
        String TicketP = "Ticket Number";

        String Ticketfare    = "http://titanic/TicketFare";
        String FareP = "Ticket Fare";

        String CabinNumber    = "http://titanic/CabinNumber";
        String CabinP = "Cabin Number";

        String Embark    = "http://titanic/Embarked";
        String EmbarkedP = "Embarked";
        //String fullName     = givenName + " " + familyName;
        // create an empty model
        Model model = ModelFactory.createDefaultModel();

        // create the resource
        //   and add the properties cascading style
        Resource sex = model.createResource(personSex).addProperty(VCARD.N, SexP);
        Resource survival = model.createResource(personSurvival).addProperty(VCARD.N, SurviveP);
        Resource pclass = model.createResource(personClass).addProperty(VCARD.N, ClassP);
        Resource Age = model.createResource(personAge).addProperty(VCARD.N, AgeP);
        Resource Siblings = model.createResource(personSib).addProperty(VCARD.N, SiblingsP);
        Resource Parents = model.createResource(personPar).addProperty(VCARD.N, ParentsP);
        Resource TicketN = model.createResource(TicketNumber).addProperty(VCARD.N, TicketP);
        Resource Fare = model.createResource(Ticketfare).addProperty(VCARD.N, FareP);
        Resource CabinN = model.createResource(CabinNumber).addProperty(VCARD.N, CabinP);
        Resource Embarked = model.createResource(Embark).addProperty(VCARD.N, EmbarkedP);
        
        // list the statements in the graph
        StmtIterator iter = model.listStatements();
        
        // print out the predicate, subject and object of each statement
        while (iter.hasNext()) {
            Statement stmt      = iter.nextStatement();         // get next statement
            Resource  subject   = stmt.getSubject();   // get the subject
            Property  predicate = stmt.getPredicate(); // get the predicate
            RDFNode   object    = stmt.getObject();    // get the object
            
            System.out.print(subject.toString());
            System.out.print(" " + predicate.toString() + " ");
            if (object instanceof Resource) {
                System.out.print(object.toString());
            } else {
                // object is a literal
                System.out.print(" \"" + object.toString() + "\"");
            }
            System.out.println(" .");
        }
    }
}